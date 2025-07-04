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

from __future__ import with_statement
import os
import sys
import inspect
if sys.platform.startswith('java'):
    from java.lang.System import getProperty

from robot.errors import DataError

from .encoding import decode_from_system
from .error import get_error_details
from .robotpath import abspath, normpath


class Importer(object):

    def __init__(self, type=None, logger=None):
        if not logger:
            from robot.output import LOGGER as logger
        self._type = type or ''
        self._logger = logger
        self._importers = (ByPathImporter(logger),
                           NonDottedImporter(logger),
                           DottedImporter(logger))
        self._by_path_importer = self._importers[0]

    def import_class_or_module(self, name, instantiate_with_args=None):
        """Imports Python class/module or Java class with given name.

        Class can either live in a module/package or be standalone Java class.
        In the former case the name is something like 'MyClass' and in the
        latter it could be 'your.package.YourLibrary'. Python classes always
        live in a module, but if the module name is exactly same as the class
        name then simple 'MyLibrary' will import a class.

        Python modules can be imported both using format 'MyModule' and
        'mymodule.submodule'.

        `name` can also be a path to the imported file/directory. In that case
        importing is done using `import_class_or_module_by_path` method.

        If `instantiate_with_args` is not None, imported classes are
        instantiated with the specified arguments automatically.
        """
        try:
            imported, source = self._import_class_or_module(name)
            self._log_import_succeeded(imported, name, source)
            return self._instantiate_if_needed(imported, instantiate_with_args)
        except DataError, err:
            self._raise_import_failed(name, err)

    def _import_class_or_module(self, name):
        for importer in self._importers:
            if importer.handles(name):
                return importer.import_(name)

    def import_class_or_module_by_path(self, path, instantiate_with_args=None):
        """Import a Python module or Java class using a file system path.

        When importing a Python file, the path must end with '.py' and the
        actual file must also exist. When importing Java classes, the path
        must end with '.java' or '.class'. The class file must exist in both
        cases and in the former case also the source file must exist.

        If `instantiate_with_args` is not None, imported classes are
        instantiated with the specified arguments automatically.
        """
        try:
            imported, source = self._by_path_importer.import_(path)
            self._log_import_succeeded(imported, imported.__name__, source)
            return self._instantiate_if_needed(imported, instantiate_with_args)
        except DataError, err:
            self._raise_import_failed(path, err)

    def _raise_import_failed(self, name, error):
        import_type = '%s ' % self._type if self._type else ''
        msg = "Importing %s'%s' failed: %s" % (import_type, name, error.message)
        if not error.details:
            raise DataError(msg)
        msg = [msg, error.details]
        msg.extend(self._get_items_in('PYTHONPATH', sys.path))
        if sys.platform.startswith('java'):
            classpath = getProperty('java.class.path').split(os.path.pathsep)
            msg.extend(self._get_items_in('CLASSPATH', classpath))
        raise DataError('\n'.join(msg))

    def _get_items_in(self, type, items):
        yield '%s:' % type
        for item in items:
            if item:
                yield '  %s' % (item if isinstance(item, unicode)
                                else decode_from_system(item))

    def _instantiate_if_needed(self, imported, args):
        if args is None:
            return imported
        if inspect.isclass(imported):
            return self._instantiate_class(imported, args)
        if args:
            raise DataError("Modules do not take arguments.")
        return imported

    def _instantiate_class(self, imported, args):
        try:
            return imported(*args)
        except:
            raise DataError('Creating instance failed: %s\n%s' % get_error_details())

    def _log_import_succeeded(self, item, name, source):
        import_type = '%s ' % self._type if self._type else ''
        item_type = 'module' if inspect.ismodule(item) else 'class'
        location = ("'%s'" % source) if source else 'unknown location'
        self._logger.info("Imported %s%s '%s' from %s."
                          % (import_type, item_type, name, location))


class _Importer(object):

    def __init__(self, logger):
        self._logger = logger

    def _import(self, name, fromlist=None, retry=True):
        try:
            try:
                return __import__(name, fromlist=fromlist)
            except ImportError:
                # Hack to support standalone Jython. For more information, see:
                # http://code.google.com/p/robotframework/issues/detail?id=515
                # http://bugs.jython.org/issue1778514
                if sys.platform.startswith('java') and fromlist and retry:
                    __import__('%s.%s' % (name, fromlist[0]))
                    return self._import(name, fromlist, retry=False)
                # Cannot use plain raise due to
                # http://ironpython.codeplex.com/workitem/32332
                raise sys.exc_type, sys.exc_value, sys.exc_traceback
        except:
            raise DataError(*get_error_details())

    def _verify_type(self, imported):
        if inspect.isclass(imported) or inspect.ismodule(imported):
            return imported
        raise DataError('Expected class or module, got <%s>.' % type(imported).__name__)

    def _get_class_from_module(self, module, name=None):
        klass = getattr(module, name or module.__name__, None)
        return klass if inspect.isclass(klass) else None

    def _get_source(self, module):
        source = getattr(module, '__file__', None)
        return abspath(source) if source else None


class ByPathImporter(_Importer):
    _valid_import_extensions = ('.py', '.java', '.class', '')

    def handles(self, path):
        return os.path.isabs(path)

    def import_(self, path):
        self._verify_import_path(path)
        self._remove_wrong_module_from_sys_modules(path)
        module = self._import_by_path(path)
        imported = self._get_class_from_module(module) or module
        return self._verify_type(imported), path

    def _verify_import_path(self, path):
        if not os.path.exists(path):
            raise DataError('File or directory does not exist.')
        if not os.path.isabs(path):
            raise DataError('Import path must be absolute.')
        if not os.path.splitext(path)[1] in self._valid_import_extensions:
            raise DataError('Not a valid file or directory to import.')

    def _remove_wrong_module_from_sys_modules(self, path):
        importing_from, name = self._split_path_to_module(path)
        importing_package = os.path.splitext(path)[1] == ''
        if self._wrong_module_imported(name, importing_from, importing_package):
            del sys.modules[name]
            self._logger.info("Removed module '%s' from sys.modules to import "
                              "fresh module." % name)

    def _split_path_to_module(self, path):
        module_dir, module_file = os.path.split(abspath(path))
        module_name = os.path.splitext(module_file)[0]
        if module_name.endswith('$py'):
            module_name = module_name[:-3]
        return module_dir, module_name

    def _wrong_module_imported(self, name, importing_from, importing_package):
        module = sys.modules.get(name)
        if not module:
            return False
        source = getattr(module, '__file__', None)
        if not source:  # play safe (oclwrs at least with java based modules)
            return True
        imported_from, imported_package = self._get_import_information(source)
        return ((normpath(importing_from), importing_package) !=
                (normpath(imported_from), imported_package))

    def _get_import_information(self, source):
        imported_from, imported_file = self._split_path_to_module(source)
        imported_package = imported_file == '__init__'
        if imported_package:
            imported_from = os.path.dirname(imported_from)
        return imported_from, imported_package

    def _import_by_path(self, path):
        module_dir, module_name = self._split_path_to_module(path)
        sys.path.insert(0, module_dir)
        try:
            return self._import(module_name)
        finally:
            sys.path.pop(0)


class NonDottedImporter(_Importer):

    def handles(self, name):
        return '.' not in name

    def import_(self, name):
        module = self._import(name)
        imported = self._get_class_from_module(module) or module
        return self._verify_type(imported), self._get_source(module)


class DottedImporter(_Importer):

    def handles(self, name):
        return '.' in name

    def import_(self, name):
        parent_name, lib_name = name.rsplit('.', 1)
        parent = self._import(parent_name, fromlist=[str(lib_name)])
        try:
            imported = getattr(parent, lib_name)
        except AttributeError:
            raise DataError("Module '%s' does not contain '%s'."
                            % (parent_name, lib_name))
        imported = self._get_class_from_module(imported, lib_name) or imported
        return self._verify_type(imported), self._get_source(parent)
