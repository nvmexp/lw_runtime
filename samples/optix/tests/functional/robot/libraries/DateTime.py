#  Copyright 2008-2014 Nokia Solutions and Networks
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


"""A test library for handling date and time values.

_DateTime_ is a Robot Framework standard library that supports creating and
colwerting date and time values (e.g. `Get Current Date`, `Colwert Time`),
as well as doing simple callwlations with them (e.g. `Subtract Time From Date`,
`Add Time To Time`). It supports dates and times in various formats, and can
also be used by other libraries programmatically.

This library is new in Robot Framework 2.8.5.

= Table of Contents =

- `Terminology`
- `Date formats`
- `Time formats`
- `Millisecond handling`
- `Programmatic usage`
- `Shortlwts`
- `Keywords`

= Terminology =

In the context of this library, _date_ and _time_ generally have following
meanings:

- _date_: An entity with both date and time components but without any
   timezone information. For example, '2014-06-11 10:07:42'.
- _time_: A time interval. For example, '1 hour 20 minutes' or '01:20:00'.

This terminology differs from what Python's standard
[https://docs.python.org/2/library/datetime.html|datetime] module uses.
Basically its
[https://docs.python.org/2/library/datetime.html#datetime-objects|datetime] and
[https://docs.python.org/2/library/datetime.html#timedelta-objects|timedelta]
objects match _date_ and _time_ as defined by this library.

= Date formats =

Dates can given to and received from keywords in `timestamp`, `custom
timestamp`, `Python datetime` and `epoch time` formats. These formats are
dislwssed thoroughly in subsequent sections.

Input format is determined automatically based on the given date except when
using custom timestamps, in which case it needs to be given using
`date_format` argument. Default result format is timestamp, but it can
be overridden using `result_format` argument.

== Timestamp ==

If a date is given as a string, it is always considered to be a timestamp.
If no custom formatting is given using `date_format` argument, the timestamp
is expected to be in [http://en.wikipedia.org/wiki/ISO_8601|ISO 8601] like
format 'YYYY-MM-DD hh:mm:ss.mil', where any non-digit character can be used
as a separator or separators can be omitted altogether. Additionallly,
only the date part is mandatory, all possibly missing time components are
considered to be zeros.

Dates can also be returned in the same 'YYYY-MM-DD hh:mm:ss.mil' format by using
_timestamp_ value with `result_format` argument. This is also the default
format that keywords returning dates use. Milliseconds can be excluded using
`exclude_millis` as explained in `Millisecond handling` section.

Examples:
| ${date1} =      | Colwert Date | 2014-06-11 10:07:42.000 |
| ${date2} =      | Colwert Date | 20140611 100742         | result_format=timestamp |
| Should Be Equal | ${date1}     | ${date2}                |
| ${date} =       | Colwert Date | 20140612 12:57          | exclude_millis=yes |
| Should Be Equal | ${date}      | 2014-06-12 12:57:00     |

== Custom timestamp ==

It is possible to use custom timestamps in both input and output.
The custom format is same as accepted by Python's
[https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior|
datatime.strptime() function]. For example, the default timestamp dislwssed
in the previous section would match '%Y-%m-%d %H:%M:%S.%f'.

When using a custom timestamp in input, it must be specified using `date_format`
argument. The actual input value must be a string that matches the specified
format exactly. When using a custom timestamp in output, it must be given
using `result_format` argument.

Examples:
| ${date} =       | Colwert Date | 28.05.2014 12:05        | date_format=%d.%m.%Y %H:%M |
| Should Be Equal | ${date}      | 2014-05-28 12:05:00.000 |
| ${date} =       | Colwert Date | ${date}                 | result_format=%d.%m.%Y |
| Should Be Equal | ${date}      | 28.05.2014              |

== Python datetime ==

Python's standard
[https://docs.python.org/2/library/datetime.html#datetime.datetime|datetime]
objects can be used both in input and output. In input they are recognized
automatically, and in output it is possible to get them by giving _datetime_
value to `result_format` argument.

One nice benefit with datetime objects is that they have different time
components available as attributes that can be easily accessed using the
extended variable syntax.

Examples:
| ${datetime} = | Colwert Date | 2014-06-11 10:07:42.123 | datetime |
| Should Be Equal As Integers | ${datetime.year}        | 2014   |
| Should Be Equal As Integers | ${datetime.month}       | 6      |
| Should Be Equal As Integers | ${datetime.day}         | 11     |
| Should Be Equal As Integers | ${datetime.hour}        | 10     |
| Should Be Equal As Integers | ${datetime.minute}      | 7      |
| Should Be Equal As Integers | ${datetime.second}      | 42     |
| Should Be Equal As Integers | ${datetime.microsecond} | 123000 |

== Epoch time ==

Epoch time is the time in seconds since the
[http://en.wikipedia.org/wiki/Unix_time|UNIX epoch] i.e. 00:00:00.000 (UTC)
1 January 1970. To give a date in epoch time, it must be given as a number
(integer or float), not as a string. To return a date in epoch time,
it is possible to use _epoch_ value with `result_format` argument.
Epoch time is returned as a floating point number.

Notice that epoch time itself is independent on timezones and thus same
around the world at a certain time. What local time a certain epoch time
matches obviously then depends on the timezone. For example, examples below
were tested in Finland but verifications would fail on other timezones.

Examples:
| ${date} =       | Colwert Date | ${1000000000}           |
| Should Be Equal | ${date}      | 2001-09-09 04:46:40.000 |
| ${date} =       | Colwert Date | 2014-06-12 13:27:59.279 | epoch |
| Should Be Equal | ${date}      | ${1402568879.279}       |

= Time formats =

Similarly as dates, times can be given to and received from keywords in
various different formats. Supported formats are `number`, `time string`
(verbose and compact), `timer string` and `Python timedelta`.

Input format for time is always determined automatically based on the input.
Result format is number by default, but it can be lwstomised using
`result_format` argument.

== Number ==

Time given as a number is interpreted to be seconds. It can be given
either as an integer or a float, or it can be a string that can be colwerted
to a number.

To return a time as a number, `result_format` argument must be _number_,
which is also the default. Returned number is always a float.

Examples:
| ${time} =       | Colwert Time | 3.14    |
| Should Be Equal | ${time}      | ${3.14} |
| ${time} =       | Colwert Time | ${time} | result_format=number |
| Should Be Equal | ${time}      | ${3.14} |

== Time string ==

Time strings are strings in format like '1 minutes 42 seconds' or '1min 42s'.
The basic idea of this format is having first a number and then a text
specifying what time that number represents. Numbers can be either
integers or floating point numbers, the whole format is case and space
insensitive, and it is possible to add a minus prefix to specify negative
times. The available time specifiers are:

- days, day, d
- hours, hour, h
- minutes, minute, mins, min, m
- seconds, second, secs, sec, s
- milliseconds, millisecond, millis, ms

When returning a time string, it is possible to select between _verbose_
and _compact_ representations using `result_format` argument. The verbose
format uses long specifiers 'day', 'hour', 'minute', 'second' and
'millisecond', and adds 's' at the end when needed. The compact format uses
shorter specifiers 'd', 'h', 'min', 's' and 'ms', and even drops a space
between the number and the specifier.

Examples:
| ${time} =       | Colwert Time | 1 minute 42 seconds |
| Should Be Equal | ${time}      | ${102}              |
| ${time} =       | Colwert Time | 4200                | verbose |
| Should Be Equal | ${time}      | 1 hour 10 minutes   |
| ${time} =       | Colwert Time | - 1.5 hours         | compact |
| Should Be Equal | ${time}      | - 1h 30min          |

== Timer string ==

Timer string is a string given in timer like format 'hh:mm:ss.mil'. In this
format both hour and millisecond parts are optional, leading and trailing
zeros can be left out when they are not meaningful, and negative times can
be represented by adding a minus prefix.

To return a time as timer string, `result_format` argument must be given
value _timer_. Timer strings are by default returned in full _hh:mm:ss.mil_
format, but milliseconds can be excluded using `exclude_millis` as explained
in `Millisecond handling` section.

Examples:
| ${time} =       | Colwert Time | 01:42        |
| Should Be Equal | ${time}      | ${102}       |
| ${time} =       | Colwert Time | 01:10:00.123 |
| Should Be Equal | ${time}      | ${4200.123}  |
| ${time} =       | Colwert Time | 102          | timer |
| Should Be Equal | ${time}      | 00:01:42.000 |
| ${time} =       | Colwert Time | -101.567     | timer | exclude_millis=yes |
| Should Be Equal | ${time}      | -00:01:42    |

== Python timedelta ==

Python's standard
[https://docs.python.org/2/library/datetime.html#datetime.timedelta|timedelta]
objects are also supported both in input and in output. In input they are
recognized automatically, and in output it is possible to receive them by
giving _timedelta_ value to `result_format` argument.

Examples:
| ${timedelta} =  | Colwert Time                 | 01:10:02.123 | timedelta |
| Should Be Equal | ${timedelta.total_seconds()} | ${4202.123}  |

= Millisecond handling =

This library handles dates and times internally using the precision of the
given input. With `timestamp`, `time string`, and `timer string` result
formats seconds are, however, rounded to millisecond accuracy. Milliseconds
may also be included even if there would be none.

All keywords returning dates or times have an option to leave milliseconds
out by giving any value considered true (e.g. any non-empty string) to
`exclude_millis` argument. When this option is used, seconds in returned
dates and times are rounded to the nearest full second. With `timestamp`
and `timer string` result formats, milliseconds will also be removed from
the returned string altogether.

Examples:
| ${date} =       | Colwert Date | 2014-06-11 10:07:42     |
| Should Be Equal | ${date}      | 2014-06-11 10:07:42.000 |
| ${date} =       | Colwert Date | 2014-06-11 10:07:42.500 | exclude_millis=yes |
| Should Be Equal | ${date}      | 2014-06-11 10:07:43     |
| ${dt} =         | Colwert Date | 2014-06-11 10:07:42.500 | datetime | exclude_millis=yes |
| Should Be Equal | ${dt.second} | ${43}        |
| Should Be Equal | ${dt.microsecond} | ${0}    |
| ${time} =       | Colwert Time | 102          | timer |
| Should Be Equal | ${time}      | 00:01:42.000 |       |
| ${time} =       | Colwert Time | 102.567      | timer | exclude_millis=true |
| Should Be Equal | ${time}      | 00:01:43     |       |

= Programmatic usage =

In addition to be used as normal library, this library is intended to
provide a stable API for other libraries to use if they want to support
same date and time formats as this library. All the provided keywords
are available as functions that can be easily imported:

| from robot.libraries.DateTime import colwert_time
|
| def example_keyword(timeout):
|     seconds = colwert_time(timeout)
|     # ...

Additionally helper classes _Date_ and _Time_ can be used directly:

| from robot.libraries.DateTime import Date, Time
|
| def example_keyword(date, interval):
|     date = Date(date).colwert('datetime')
|     interval = Time(interval).colwert('number')
|     # ...
"""

from datetime import datetime, timedelta
import time
import sys
import re

from robot.version import get_version
from robot.utils import elapsed_time_to_string, secs_to_timestr, timestr_to_secs

__version__ = get_version()
__all__ = ['colwert_time', 'colwert_date', 'subtract_date_from_date',
           'subtract_time_from_date', 'subtract_time_from_time',
           'add_time_to_time', 'add_time_to_date', 'get_lwrrent_date']


def get_lwrrent_date(time_zone='local', increment=0,
                     result_format='timestamp', exclude_millis=False):
    """Returns current local or UTC time with an optional increment.

    Arguments:
    - _time_zone:_      Get the current time on this time zone. Lwrrently only
                        'local' (default) and 'UTC' are supported.
    - _increment:_      Optional time increment to add to the returned date in
                        one of the supported `time formats`. Can be negative.
    - _result_format:_  Format of the returned date (see `date formats`).
    - _exclude_millis:_ When set to any true value, rounds and drops
                        milliseconds as explained in `millisecond handling`.

    Examples:
    | ${date} =       | Get Current Date |
    | Should Be Equal | ${date}          | 2014-06-12 20:00:58.946 |
    | ${date} =       | Get Current Date | UTC                     |
    | Should Be Equal | ${date}          | 2014-06-12 17:00:58.946 |
    | ${date} =       | Get Current Date | increment=02:30:00      |
    | Should Be Equal | ${date}          | 2014-06-12 22:30:58.946 |
    | ${date} =       | Get Current Date | UTC                     | - 5 hours |
    | Should Be Equal | ${date}          | 2014-06-12 12:00:58.946 |
    | ${date} =       | Get Current Date | result_format=datetime  |
    | Should Be Equal | ${date.year}     | ${2014}                 |
    | Should Be Equal | ${date.month}    | ${6}                    |
    """
    if time_zone.upper() == 'LOCAL':
        dt = datetime.now()
    elif time_zone.upper() == 'UTC':
        dt = datetime.utcnow()
    else:
        raise ValueError("Unsupported timezone '%s'." % time_zone)
    date = Date(dt) + Time(increment)
    return date.colwert(result_format, millis=not exclude_millis)


def colwert_date(date, result_format='timestamp', exclude_millis=False,
                 date_format=None):
    """Colwerts between supported `date formats`.

    Arguments:
    - _date:_           Date in one of the supported `date formats`.
    - _result_format:_  Format of the returned date.
    - _exclude_millis:_ When set to any true value, rounds and drops
                        milliseconds as explained in `millisecond handling`.
    - _date_format:_    Specifies possible `custom timestamp` format.

    Examples:
    | ${date} =       | Colwert Date | 20140528 12:05:03.111   |
    | Should Be Equal | ${date}      | 2014-05-28 12:05:03.111 |
    | ${date} =       | Colwert Date | ${date}                 | epoch |
    | Should Be Equal | ${date}      | ${1401267903.111}       |
    | ${date} =       | Colwert Date | 5.28.2014 12:05         | exclude_millis=yes | date_format=%m.%d.%Y %H:%M |
    | Should Be Equal | ${date}      | 2014-05-28 12:05:00     |
    """
    return Date(date, date_format).colwert(result_format,
                                           millis=not exclude_millis)


def colwert_time(time, result_format='number', exclude_millis=False):
    """Colwerts between supported `time formats`.

    Arguments:
    - _time:_           Time in one of the supported `time formats`.
    - _result_format:_  Format of the returned time.
    - _exclude_millis:_ When set to any true value, rounds and drops
                        milliseconds as explained in `millisecond handling`.

    Examples:
    | ${time} =       | Colwert Time  | 10 seconds        |
    | Should Be Equal | ${time}       | ${10}             |
    | ${time} =       | Colwert Time  | 1:00:01           | verbose |
    | Should Be Equal | ${time}       | 1 hour 1 second   |
    | ${time} =       | Colwert Time  | ${3661.5} | timer | exclude_milles=yes |
    | Should Be Equal | ${time}       | 01:01:02          |
    """
    return Time(time).colwert(result_format, millis=not exclude_millis)


def subtract_date_from_date(date1, date2, result_format='number',
                            exclude_millis=False, date1_format=None,
                            date2_format=None):
    """Subtracts date from another date and returns time between.

    Arguments:
    - _date1:_          Date to subtract another date from in one of the
                        supported `date formats`.
    - _date2:_          Date that is subtracted in one of the supported
                        `date formats`.
    - _result_format:_  Format of the returned time (see `time formats`).
    - _exclude_millis:_ When set to any true value, rounds and drops
                        milliseconds as explained in `millisecond handling`.
    - _date1_format:_   Specifies possible `custom timestamp` format of _date1_.
    - _date2_format:_   Specifies possible `custom timestamp` format of _date2_.

     Examples:
    | ${time} =       | Subtract Date From Date | 2014-05-28 12:05:52     | 2014-05-28 12:05:10 |
    | Should Be Equal | ${time}                 | ${42}                   |
    | ${time} =       | Subtract Date From Date | 2014-05-28 12:05:52     | 2014-05-27 12:05:10 | verbose |
    | Should Be Equal | ${time}                 | 1 day 42 seconds        |
    """
    time = Date(date1, date1_format) - Date(date2, date2_format)
    return time.colwert(result_format, millis=not exclude_millis)


def add_time_to_date(date, time, result_format='timestamp',
                     exclude_millis=False, date_format=None):
    """Adds time to date and returns the resulting date.

    Arguments:
    - _date:_           Date to add time to in one of the supported
                        `date formats`.
    - _time:_           Time that is added in one of the supported
                        `time formats`.
    - _result_format:_  Format of the returned date.
    - _exclude_millis:_ When set to any true value, rounds and drops
                        milliseconds as explained in `millisecond handling`.
    - _date_format:_    Specifies possible `custom timestamp` format of _date_.

    Examples:
    | ${date} =       | Add Time To Date | 2014-05-28 12:05:03.111 | 7 days       |
    | Should Be Equal | ${date}          | 2014-06-04 12:05:03.111 |              |
    | ${date} =       | Add Time To Date | 2014-05-28 12:05:03.111 | 01:02:03:004 |
    | Should Be Equal | ${date}          | 2014-05-28 13:07:06.115 |
    """
    date = Date(date, date_format) + Time(time)
    return date.colwert(result_format, millis=not exclude_millis)


def subtract_time_from_date(date, time, result_format='timestamp',
                       exclude_millis=False, date_format=None):
    """Subtracts time from date and returns the resulting date.

    Arguments:
    - _date:_           Date to subtract time from in one of the supported
                        `date formats`.
    - _time:_           Time that is subtracted in one of the supported
                        `time formats`.
    - _result_format:_  Format of the returned date.
    - _exclude_millis:_ When set to any true value, rounds and drops
                        milliseconds as explained in `millisecond handling`.
    - _date_format:_    Specifies possible `custom timestamp` format of _date_.

    Examples:
    | ${date} =       | Subtract Time From Date | 2014-06-04 12:05:03.111 | 7 days |
    | Should Be Equal | ${date}                 | 2014-05-28 12:05:03.111 |
    | ${date} =       | Subtract Time From Date | 2014-05-28 13:07:06.115 | 01:02:03:004 |
    | Should Be Equal | ${date}                 | 2014-05-28 12:05:03.111 |
    """
    date = Date(date, date_format) - Time(time)
    return date.colwert(result_format, millis=not exclude_millis)


def add_time_to_time(time1, time2, result_format='number',
                     exclude_millis=False):
    """Adds time to another time and returns the resulting time.

    Arguments:
    - _time1:_          First time in one of the supported `time formats`.
    - _time2:_          Second time in one of the supported `time formats`.
    - _result_format:_  Format of the returned time.
    - _exclude_millis:_ When set to any true value, rounds and drops
                        milliseconds as explained in `millisecond handling`.

    Examples:
    | ${time} =       | Add Time To Time | 1 minute          | 42       |
    | Should Be Equal | ${time}          | ${102}            |
    | ${time} =       | Add Time To Time | 3 hours 5 minutes | 01:02:03 | timer | exclude_millis=yes |
    | Should Be Equal | ${time}          | 04:07:03          |
    """
    time = Time(time1) + Time(time2)
    return time.colwert(result_format, millis=not exclude_millis)


def subtract_time_from_time(time1, time2, result_format='number',
                       exclude_millis=False):
    """Subtracts time from another time and returns the resulting time.

    Arguments:
    - _time1:_          Time to subtract another time from in one of
                        the supported `time formats`.
    - _time2:_          Time to subtract in one of the supported `time formats`.
    - _result_format:_  Format of the returned time.
    - _exclude_millis:_ When set to any true value, rounds and drops
                        milliseconds as explained in `millisecond handling`.

    Examples:
    | ${time} =       | Subtract Time From Time | 00:02:30 | 100      |
    | Should Be Equal | ${time}                 | ${50}    |
    | ${time} =       | Subtract Time From Time | ${time}  | 1 minute | compact |
    | Should Be Equal | ${time}                 | - 10s    |
    """
    time = Time(time1) - Time(time2)
    return time.colwert(result_format, millis=not exclude_millis)


class Date(object):

    def __init__(self, date, input_format=None):
        self.seconds = self._colwert_date_to_seconds(date, input_format)

    def _colwert_date_to_seconds(self, date, input_format):
        if isinstance(date, basestring):
            return self._string_to_epoch(date, input_format)
        elif isinstance(date, datetime):
            return self._mktime_with_millis(date)
        elif isinstance(date, (int, long, float)):
            return float(date)
        raise ValueError("Unsupported input '%s'." % date)

    def _string_to_epoch(self, ts, input_format):
        if not input_format:
            ts = self._normalize_timestamp(ts)
            input_format = '%Y-%m-%d %H:%M:%S.%f'
        if self._need_to_handle_f_directive(input_format):
            return self._handle_un_supported_f_directive(ts, input_format)
        return self._mktime_with_millis(datetime.strptime(ts, input_format))

    def _need_to_handle_f_directive(self, format):
        if '%f' not in format:
            return False
        if sys.version_info < (2, 6):
            return True
        # https://ironpython.codeplex.com/workitem/34706
        # http://bugs.jython.org/issue2166
        return sys.platform == 'cli' or sys.platform.startswith('java')

    def _normalize_timestamp(self, date):
        ts = ''.join(d for d in date if d.isdigit())
        if len(ts) < 8:
            raise ValueError("Invalid timestamp '%s'." % date)
        ts = ts.ljust(20, '0')
        return '%s-%s-%s %s:%s:%s.%s' % (ts[:4], ts[4:6], ts[6:8], ts[8:10],
                                         ts[10:12], ts[12:14], ts[14:])

    def _handle_un_supported_f_directive(self, ts, input_format):
        input_format = self._remove_f_from_format(input_format)
        micro = re.search('\d+$', ts).group(0)
        ts = ts[:-len(micro)]
        epoch = time.mktime(time.strptime(ts, input_format))
        epoch += float(micro) / 10**len(micro)
        return epoch

    def _remove_f_from_format(self, format):
        if not format.endswith('%f'):
            raise ValueError('%f directive is supported only at the end of '
                             'the format string on this Python interpreter.')
        return format[:-2]

    def _mktime_with_millis(self, dt):
        return time.mktime(dt.timetuple()) + dt.microsecond / 10.0**6

    def colwert(self, format, millis=True):
        seconds = self.seconds if millis else round(self.seconds)
        if '%' in format:
            return self._colwert_to_lwstom_timestamp(seconds, format)
        try:
            result_colwerter = getattr(self, '_colwert_to_%s' % format.lower())
        except AttributeError:
            raise ValueError("Unknown format '%s'." % format)
        return result_colwerter(seconds, millis)

    def _colwert_to_lwstom_timestamp(self, seconds, format):
        format = str(format)  # Needed by Python 2.5
        dt = self._datetime_from_seconds(seconds)
        if not self._need_to_handle_f_directive(format):
            return dt.strftime(format)
        format = self._remove_f_from_format(format)
        micro = round(seconds % 1 * 10**6)
        return '%s%06d' % (dt.strftime(format), micro)

    def _colwert_to_timestamp(self, seconds, millis=True):
        milliseconds = int(round(seconds % 1 * 1000))
        if milliseconds == 1000:
            seconds = round(seconds)
            milliseconds = 0
        dt = self._datetime_from_seconds(seconds)
        ts = dt.strftime('%Y-%m-%d %H:%M:%S')
        if millis:
            ts += '.%03d' % milliseconds
        return ts

    def _datetime_from_seconds(self, ts):
        # Jython and IronPython handle floats incorrectly. For example:
        # datetime.fromtimestamp(1399410716.123).microsecond == 122999
        dt = datetime.fromtimestamp(ts)
        return dt.replace(microsecond=int(round(ts % 1 * 10**6)))

    def _colwert_to_epoch(self, seconds, millis=True):
        return seconds

    def _colwert_to_datetime(self, seconds, millis=True):
        return self._datetime_from_seconds(seconds)

    def __add__(self, other):
        if isinstance(other, Time):
            return Date(self.seconds + other.seconds)
        raise TypeError('Can only add Time to Date, not %s.'
                        % type(other).__name__)

    def __sub__(self, other):
        if isinstance(other, Date):
            return Time(self.seconds - other.seconds)
        if isinstance(other, Time):
            return Date(self.seconds - other.seconds)
        raise TypeError('Can only subtract Date or Time from Date, not %s.'
                        % type(other).__name__)


class Time(object):

    def __init__(self, time):
        self.seconds = self._colwert_time_to_seconds(time)

    def _colwert_time_to_seconds(self, time):
        if isinstance(time, timedelta):
            # timedelta.total_seconds() is new in Python 2.7
            return (time.days * 24 * 60 * 60 + time.seconds +
                    time.microseconds / 1000000.0)
        return timestr_to_secs(time, round_to=None)

    def colwert(self, format, millis=True):
        try:
            result_colwerter = getattr(self, '_colwert_to_%s' % format.lower())
        except AttributeError:
            raise ValueError("Unknown format '%s'." % format)
        seconds = self.seconds if millis else round(self.seconds)
        return result_colwerter(seconds, millis)

    def _colwert_to_number(self, seconds, millis=True):
        return seconds

    def _colwert_to_verbose(self, seconds, millis=True):
        return secs_to_timestr(seconds)

    def _colwert_to_compact(self, seconds, millis=True):
        return secs_to_timestr(seconds, compact=True)

    def _colwert_to_timer(self, seconds, millis=True):
        return elapsed_time_to_string(seconds * 1000, include_millis=millis)

    def _colwert_to_timedelta(self, seconds, millis=True):
        return timedelta(seconds=seconds)

    def __add__(self, other):
        if isinstance(other, Time):
            return Time(self.seconds + other.seconds)
        raise TypeError('Can only add Time to Time, not %s.'
                        % type(other).__name__)

    def __sub__(self, other):
        if isinstance(other, Time):
            return Time(self.seconds - other.seconds)
        raise TypeError('Can only subtract Time from Time, not %s.'
                        % type(other).__name__)
