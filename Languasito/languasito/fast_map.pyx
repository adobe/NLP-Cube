# distutils: language = c++
import cython
from libcpp.map cimport map
from libcpp.string  cimport string

cdef class FastMap:
      cdef:
            map[string, int] components

      def __setitem__(self, key, value):
            self.components[key] = value

      def __getitem__(self, key):
            return self.components[key]