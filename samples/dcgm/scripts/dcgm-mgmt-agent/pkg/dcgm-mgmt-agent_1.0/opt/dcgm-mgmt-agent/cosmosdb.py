#!/usr/bin/python

import argparse
import hashlib
import os
import sqlite3 as lite
import subprocess
import sys
import json
import fcntl
import errno
import time as _time

from datetime import datetime, time, timedelta
from time import gmtime, strftime

_SCRIPT_NAME = os.path.basename(__file__).replace('.py', '')
_DATABASE = '/opt/dcgm-mgmt-agent/db/cosmos.db'
_LOCKFILE = '/opt/dcgm-mgmt-agent/db/dblock'

def configure_parser():
        formatter = argparse.RawDescriptionHelpFormatter
        parser    = argparse.ArgumentParser(usage                 = None,
                                            fromfile_prefix_chars = '@',
                                            formatter_class       = formatter,
                                            description           = __doc__,
                                            prog                  = _SCRIPT_NAME)
        parser.add_argument('-a', '--all',
                            action = 'store_true',
                            help = 'Print the list of all keys & values.')
        parser.add_argument('-j', '--json',
                            action = 'store_true',
                            help = 'Print the output in json format.')
        parser.add_argument('-k', '--key',
                            help = 'The key that needs to be read or written')
        parser.add_argument('-v', '--value',
                            help = 'If specified, value will be stored in the database.')
        parser.add_argument('-l', '--log',
                            help = 'Log an important event into the database')
        parser.add_argument('-s', '--show-logs',
        	                action = 'store_true',
                            help = 'Display all the logs')
        return parser


def main(args):
        parser = configure_parser()
        options = parser.parse_args(args)

        if (options.all == True and options.log != None):
            print "ERROR: '--all' and '--log' cannot be used at the same time";
            return;

        if (options.all == True and options.key != None):
            print "ERROR: '--all' and '--key' cannot be used at the same time";
            return;

        if (options.key == None and options.value != None):
        	print "ERROR: '--key' needs to be specified";
        	return;

        try:
                dbfile = _DATABASE;
                db = Database(dbfile);
                work(db, options)
        except IOError as e:
                print "*** " + str(e)
                return 1
        except RuntimeError as e:
                print "*** " + str(e)
                return 1
	except lite.Error as e:
		print "Error %s:" % e.args[0]
                raise
        return 0

class gbl:
        now = datetime.now()
        adds = 0
        dups = 0
        perrors = 0


### Database Class that takes care of all the node config info
class Database(object):
        def __init__(self, dbfile, verbose=False):
                # Open the database
                self.dbfile = dbfile
                self.verbose = verbose

                # TODO: Uncomment the acquire lock once we go to production
                #self.acquireLock();
                self.con = lite.connect(dbfile)
                self.lwr = self.con.cursor()
                self.lwr.execute(("CREATE TABLE IF NOT EXISTS config ("
                                  "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                                  "  key TEXT,"
                                  "  value TEXT,"
                                  "  UNIQUE(key))"))
                self.lwr.execute("CREATE TABLE IF NOT EXISTS log (id INTEGER PRIMARY KEY AUTOINCREMENT, tdate TEXT, description TEXT)")

        def acquireLock(self):
            self.lock = open(_LOCKFILE, 'w+')

            # This part ensures that the process waits until the lock is released
            while True:
                try:
                    fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB);
                    return
                except IOError as e:
                    # raise on unrelated IOErrors
                    if e.errno != errno.EAGAIN:
                        raise
                else:
                    _time.sleep(0.1)

        # Helper function for logging.  Inserts a row into log table.
        def log(self, msg, also_print=False):
                tdate = gbl.now.strftime("%b %d %Y %H:%M:%S")
                self.lwr.execute("INSERT INTO log (tdate, description) values(:1, :2)", [tdate, msg])

        def updateKey(self, key, value):
                self.lwr.execute("SELECT EXISTS(SELECT 1 FROM config WHERE key=? LIMIT 1)", (key,));
                data =  self.lwr.fetchall();
                if ((data[0])[0] == 0):
                    self.lwr.execute("INSERT INTO config (key, value) values (?, ?)", (key,value,));
                    self.log("Adding key   : "+key+" = "+value);
                else:
                    self.lwr.execute("UPDATE config set value=? where key=?", (value, key,));
                    self.log("Updating key : "+key+" = "+value);

        def showLogs(self):
                self.lwr.execute("select * from log order by id asc");
                data =  self.lwr.fetchall();
                for lg in data:
                    print lg[1], ": ", lg[2];

        def readKey(self, key, jflag=False):
                arr = {};
                self.lwr.execute("select value from config where key=?", (key,));
                data =  self.lwr.fetchall();
                for lg in data:
                    if (jflag == False):
                        print lg[0];
                    else:
                        arr[key] = lg[0];
                if (jflag == True):
                    print json.dumps(arr);

        def showAllKeys(self, jflag=False):
                arr = {};
                self.lwr.execute("select key,value from config");
                data =  self.lwr.fetchall();
                for lg in data:
                    if (jflag == False):
                        print lg[0], lg[1];
                    else:
                        arr[lg[0]] = lg[1];

                if (jflag == True):
                    print json.dumps(arr);


        def __del__(self):
                self.con.commit()
                self.con.close()
                # TODO: Uncomment htis once we get to production
                #fcntl.flock(self.lock, fcntl.LOCK_UN)

### Main routine that evaluates all the options ###
def work(db, options):

	# Write if the user has specified both key and value
	if (options.key != None and options.value != None):
		db.updateKey(options.key, options.value);
	if (options.key != None and options.value == None):
		db.readKey(options.key, options.json);
	
	if (options.log != None):
		db.log(options.log);

	if (options.show_logs == True):
		db.showLogs();

	if (options.all == True):
		db.showAllKeys(options.json);

### Main entry ###
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
