# -*- coding: utf-8 -*-

#Modified version of https://github.com/jkoelker/influxdb-collectd-write/blob/master/write_influxdb.py
#Modified by LWPU

# THIS FILE IS NO LONGER MAINTAINED, and THE LATEST VERSION OF THIS FILE CAN BE OBTAINED FROM GITHUB-COSMOS
# **** ANY EDITS IN THIS FILE WILL BE OVERWRITTEN ****

import monotime
import socket
import collections
import Queue as queue
import time
import threading
import traceback
import collectd
import influxdb
import json
import requests
import datetime
from subprocess import check_output

STATUS_FILE='/tmp/collectd_write_status'
# Interval in seconds after which cluster_id will be read from database
CL_ID_REFRESH_INTERVAL = 600

def parse_types_file(path):
    """ This function tries to parse a collectd compliant types.db file.
    Basically stolen from collectd-carbon/memsql-collectd.
    """
    data = {}

    collectd.info("Reading %s" % path)

    f = open(path, 'r')

    for line in f:
        fields = line.split()
        if len(fields) < 2:
            continue

        type_name = fields[0]

        if type_name[0] == '#':
            continue

        v = []
        for ds in fields[1:]:
            ds = ds.rstrip(',')
            ds_fields = ds.split(':')

            if len(ds_fields) != 4:
                continue

            v.append(ds_fields)

        data[type_name] = v

    f.close()

    return data


def parse_types(*paths):
    data = {}

    for path in paths:
        try:
            data.update(parse_types_file(path))
        except IOError:
            pass

    return data


def str_to_num(s):
    """
    Colwert type limits from strings to floats for arithmetic.
    Will force U[nlimited] values to be 0.
    """

    try:
        n = float(s)
    except ValueError:
        n = 0

    return n


def format_identifier(value):
    plugin_name = value.plugin
    type_name = value.type

    if value.plugin_instance:
        plugin_name = '%s-%s' % (plugin_name, value.plugin_instance)

    if value.type_instance:
        type_name = '%s-%s' % (type_name, value.type_instance)

    return '%s/%s/%s' % (value.host, plugin_name, type_name)


def PeriodicTimer(interval, function, *args, **kwargs):
    return _PeriodicTimer(interval, function, args, kwargs)


class _PeriodicTimer(threading._Timer):
    def run(self):
        while not self.finished.is_set():
            self.finished.wait(self.interval)
            if not self.finished.is_set():
                try:
                    self.function(*self.args, **self.kwargs)
                except:
                    collectd.error(traceback.format_exc())


class BulkPriorityQueue(queue.PriorityQueue):
    def put(self, item, *args, **kwargs):
        return queue.PriorityQueue.put(self, (monotime.monotonic(), item),
                                       *args, **kwargs)

    def get_bulk(self, timeout=-1, size=0, flush=False):
        values = []
        add = values.append

        if timeout < 0:
            timeout = 0

        now = monotime.monotonic()
        timeout = now - timeout

        self.not_empty.acquire()
        try:
            while self._qsize():
                if (flush or self.queue[0][0] < timeout or
                        self._qsize() > size):
                    add(self._get()[1])

                else:
                    break

            if values:
                self.not_full.notify()

            return values
        finally:
            self.not_empty.release()

class InfluxDBSSLClient(influxdb.InfluxDBClient):
    def __init__(self,
             host='localhost',
             port=8086,
             username='root',
             password='root',
             database=None,
             ssl=False,
             verify_ssl=False,
             timeout=None,
             use_udp=False,
             udp_port=4444,
             proxies=None,
             cert=None,
             key=None,
             ):
        """Construct a new InfluxDBSSLClient object."""
        self.__host = host
        self.__port = int(port)
        self._username = username
        self._password = password
        self._database = database
        self._timeout = timeout
        self._cert = cert
        self._key = key

        self._verify_ssl = verify_ssl

        self.use_udp = use_udp
        self.udp_port = udp_port
        self._session = requests.Session()
        if use_udp:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._scheme = "http"

        if ssl is True:
            self._scheme = "https"

        if proxies is None:
            self._proxies = {}
        else:
            self._proxies = proxies

        self.__baseurl = "{0}://{1}:{2}".format(
            self._scheme,
            self._host,
            self._port)

        self._headers = {
            'Content-type': 'application/json',
            'Accept': 'text/plain'
        }

    @property
    def _baseurl(self):
        return self._get_baseurl()

    def _get_baseurl(self):
        return self.__baseurl

    @property
    def _host(self):
        return self._get_host()

    def _get_host(self):
        return self.__host

    @property
    def _port(self):
        return self._get_port()

    def _get_port(self):
        return self.__port

    def request(self, url, method='GET', params=None, data=None,
                expected_response_code=200, headers=None):

        url = "{0}/{1}".format(self._baseurl, url)

        if headers is None:
            headers = self._headers

        if params is None:
            params = {}

        if isinstance(data, (dict, list)):
            data = json.dumps(data)

        # Try to send the request a maximum of three times. (see #103)
        # TODO (aviau): Make this configurable.
        for i in range(0, 3):
            try:
                if self._cert and self._key:
                    response = self._session.request(
                        method=method,
                        url=url,
                        auth=(self._username, self._password),
                        params=params,
                        data=data,
                        headers=headers,
                        proxies=self._proxies,
                        verify=self._verify_ssl,
                        timeout=self._timeout,
                        cert=(self._cert, self._key)
                    )
                else:
                    response = self._session.request(
                        method=method,
                        url=url,
                        auth=(self._username, self._password),
                        params=params,
                        data=data,
                        headers=headers,
                        proxies=self._proxies,
                        verify=self._verify_ssl,
                        timeout=self._timeout
                    )
                break
            except requests.exceptions.ConnectionError as e:
                if i < 2:
                    continue
                else:
                    with open(STATUS_FILE, 'w') as f:
                        f.write('BAD')
                    collectd.info("Connection Error: {}".format(str(e)))
                    raise e

        if response.status_code >= 500 and response.status_code < 600:
            with open(STATUS_FILE, 'w') as f:
                f.write('BAD')
            raise influxdb.exceptions.InfluxDBServerError(response.content)
        elif response.status_code == expected_response_code:
            with open(STATUS_FILE, 'w') as f:
                f.write('OK')
            return response
        else:
            with open(STATUS_FILE, 'w') as f:
                f.write('BAD')
            raise influxdb.exceptions.InfluxDBClientError(response.content, response.status_code)

class InfluxDB(object):
    def __init__(self):
        self._config = {'host': 'localhost',
                        'port': 8086,
                        'username': 'root',
                        'password': 'root',
                        'database': 'collectd',
                        'ssl': False,
                        'verify_ssl': False,
                        'cert': None,
                        'key': None,
                        'timeout': None,
                        'use_udp': False,
                        'udp_port': 4444}
        self._client = None
        self._retry = False
        self._buffer = False
        self._buffer_size = 1024
        self._buffer_sec = 10.0
        self._typesdb = ['/usr/share/collectd/types.db']
        self._types = None
        self._queues = None
        self._last_sample = {}
        self._flush_thread = None
        self._raw_values = False

    def _flush(self, timeout=-1, identifier=None, flush=False):
        if not self._buffer:
            flush = True

        if identifier:
            if identifier in self._queues:
                queues = [(identifier, self._queues[identifier])]

            else:
                queues = []

        else:
            queues = self._queues.items()

        if not flush and timeout == -1:
            if sum([q[1].qsize() for q in queues]) < self._buffer_size:
                return

        data = {}
        values = []
        add = values.extend

        for identifier, value_queue in queues:
            queue_values = value_queue.get_bulk(timeout=timeout,
                                                flush=flush)
            if not queue_values:
                #Don't busy wait
                time.sleep(0.010)
                continue

            data[identifier] = queue_values
            add(queue_values)

        try:
            #for value in values:
            #    print str(values)
            writeSt = self._client.write_points(values)
            if not writeSt:
                print "Write failed"

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError):
            if self._retry:
                for identifier, values in data.iteritems():
                    for v in values:
                        self._queues[identifier].put(v)

    def config(self, conf):
        for node in conf.children:
            key = node.key.lower()
            values = node.values

            if key in self._config:
                if key in ('ssl', 'verify_ssl', 'use_udp'):
                    self._config[key] = True

                elif key in ('port', 'timeout', 'udp_port'):
                    self._config[key] = int(values[0])

                else:
                    self._config[key] = values[0]

            elif key == 'retry':
                self._retry = True

            elif key == 'raw_values':
                self._raw_values = True

            elif key == 'buffer':
                self._buffer = values[0]
                num_values = len(values)

                if num_values == 2:
                    self._buffer_size = int(values[1])

                elif num_values == 3:
                    self._buffer_size = int(values[1])
                    self._buffer_sec = float(values[2])

            elif key == 'typesdb':
                self._typesdb.append(values[0])

    def flush(self, timeout=-1, identifier=None):
        self._flush(timeout=timeout, identifier=identifier)

    def init(self):
        self._types = parse_types(*self._typesdb)
        self._client = InfluxDBSSLClient(**self._config)
        self._queues = collections.defaultdict(lambda: BulkPriorityQueue())
        self._flush_thread = PeriodicTimer(self._buffer_sec,
                                           self._flush,
                                           flush=True)
        self._flush_thread.start()

    def shutdown(self):
        if self._flush_thread is not None:
            self._flush_thread.cancel()
            self._flush_thread.join()

        self._flush(flush=True)

    def write(self, sample):

        global cl_id_updated_time
        global cl_id

        type_info = self._types.get(sample.type)

        #print "Sample: " + str(sample)
        #print 'type_info: ' + str(type_info)

        if type_info is None:
            msg = 'plugin: %s unknown type %s, not listed in %s'

            collectd.info('write_influxdb: ' + msg % (sample.plugin,
                                                      sample.type,
                                                      self._typesdb))
            return

        secSince1970 = int(sample.time)
        usecRemainder = int(1000000.0 * (sample.time - float(secSince1970)))
        timeObj = datetime.datetime.fromtimestamp(secSince1970)
        timeObj += datetime.timedelta(microseconds=usecRemainder)
        #print "time %s, secSince1970 %s, timeObj %s" % (str(sample.time), str(secSince1970), str(timeObj))

        sampleData = {}
        sampleData['tags'] = {}
        #sampleData['time'] = timeObj #Allow the server to assign timestamps since sample.time is in localtime and influxdb is in UTC
        sampleData['fields'] = {}

        identifier = format_identifier(sample)

        dbMeasurements = {}

        cl_id_updated_time = datetime.datetime.now()

        for i, (ds_name, ds_type, min_val, max_val) in enumerate(type_info):
            value = sample.values[i]
            #colwert types to float for simplicity
            value = float(value)

            dbValueName = sample.type + '_' + ds_name

            if not isinstance(value, (float, int)):
               collectd.info("write_influxdb: ds_name %s requires float or int type. Is %s" % (ds_name, str(type(value))))
               continue

            if ds_type == "GAUGE":
                #Just store the value for gauges
                dbMeasurements[dbValueName] = value
                continue

            metric_identifier = identifier + ds_name
            last = self._last_sample.get(metric_identifier)
            lwrr_time = monotime.monotonic()
            self._last_sample[metric_identifier] = (lwrr_time, value)
            if not last:
                #collectd.info("No last yet for metric %s, measurement %s" % (ds_name, sample.type))
                continue

            old_time, old_value = last
            # Determine time between datapoints
            interval = lwrr_time - old_time
            if interval < 1:
                interval = 1

            if ds_type == "COUNTER" or ds_type == "DERIVE":
                # Check for overflow if it's a counter
                if ds_type == "COUNTER" and value < old_value:
                    if max_val == 'U':
                        # this is funky. pretend as if this is the first data
                        # point
                        new_value = None
                    else:
                        min_val = str_to_num(min_val)
                        max_val = str_to_num(max_val)
                        new_value = max_val - old_value + value - min_val
                else:
                    new_value = value - old_value

                # Both COUNTER and DERIVE get divided by the timespan
                new_value /= interval
            elif ds_type == "ABSOLUTE":
                new_value = value / interval
            else:
                collectd.warn('unrecognized ds_type {}'.format(ds_type))
                new_value = value

            dbMeasurements[dbValueName] = new_value

        sampleData['tags']['host'] = node_name
        sampleData['tags']['type'] = sample.type

        cl_id_lwrr_time = datetime.datetime.now()
        cl_id_elapsed_time = cl_id_lwrr_time - cl_id_updated_time

        # Read a fresh clusterId after every n seconds as configured in CL_ID_REFRESH_INTERVAL.
        if cl_id_elapsed_time.total_seconds() > CL_ID_REFRESH_INTERVAL:
            cl_id = check_output(['python','/usr/sbin/cosmos/cosmos-mgmt-agent/cosmosdb.py','-k','cluster_id']).rstrip('\n')
            cl_id_updated_time = datetime.datetime.now()

        if cl_id == '':
            cl_id = 'unknown'

        sampleData['tags']['cluster_id'] = cl_id

        if sample.plugin_instance:
            sampleData['tags']['plugin_instance'] = sample.plugin_instance

        if sample.type_instance:
            sampleData['tags']['type_instance'] = sample.type_instance

        if len(dbMeasurements.keys()) == 0:
            #collectd.info("Skipping metric %s with no values" % sample.type)
            return

        for dbMeasurementName in dbMeasurements.keys():
            newData = sampleData.copy()
            newData['fields'] = sampleData['fields'].copy()
            newData['measurement'] = dbMeasurementName
            newData['fields']['value'] = dbMeasurements[dbMeasurementName]
            self._queues[identifier].put(newData)
            #print "Inserting: %s" % str(newData)

        self._flush()


node_name = check_output(['python','/usr/sbin/cosmos/cosmos-mgmt-agent/cosmosdb.py','-k','node_name']).rstrip('\n')

# Cache the value of clusterId and the time when it was loaded
cl_id = check_output(['python','/usr/sbin/cosmos/cosmos-mgmt-agent/cosmosdb.py','-k','cluster_id']).rstrip('\n')
cl_id_updated_time = datetime.datetime.now()

db = InfluxDB()
collectd.register_config(db.config)
collectd.register_flush(db.flush)
collectd.register_init(db.init)
collectd.register_shutdown(db.shutdown)
collectd.register_write(db.write)
