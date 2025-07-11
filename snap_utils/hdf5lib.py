#HDF5 library wrapper for:
#*PyTables (http://www.pytables.org)
#*h5py (http://code.google.com/p/h5py/)
#
#two modes are possible:
#
#
#1. do not specifiy hdf5 interface:
#
#import hdf5lib
#
#----> this will try first tables, and if it fails will try h5py
#
#
#2. specify hdf5 interface:
#
#import hdf5lib_param
#hdf5lib_param.setlib("h5py")
#import hdf5lib
#import snapHDF5
#print hdf5lib.h5py.version
#
#or
#
#import hdf5lib_param
#hdf5lib_param.setlib("tables")
#import hdf5lib
#import snapHDF5
#print hdf5lib.tables.__version__
#
#----> this will load the specified interface; modules loaded later (like snapHDF5.py, etc.) will then use the specified interface
#
#
# Mark Vogelsberger (mvogelsb@cfa.harvard.edu)

import os
import sys
import subprocess
from pathlib import Path

repo_root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()

sys.path.append(repo_root)

import sys
import snap_utils.hdf5lib_param as hdf5lib_param

try:
	hdf5libname = hdf5lib_param.libname
	if (hdf5libname=="tables"):
		import tables
		use_tables=True
	if (hdf5libname=="h5py"):
		import h5py
		use_tables=False
except:
	try:
		import tables
		use_tables=True
	except ImportError:
		import h5py
		use_tables=False


def OpenFile(fname, mode = "r"):
	if (use_tables):
		return tables.open_file(fname, mode = mode)  #old: openFile()
	else:
		return h5py.File(fname, mode)

def GetData(f, dname):
	if (use_tables):
		return f.root._f_get_child(dname)  #old: _f_getChild()
	else:
		return f[dname]

def GetGroup(f, gname):
	if (use_tables):
		return f.root._f_get_child(gname)  #old: _f_get_child()
	else:
		return f[gname]

def GetAttr(f, gname, aname):
	if (use_tables):
		return f.root._f_get_child(gname)._f_getattr(aname)  #old: _f_getChild()  _f_getAttr()
	else:
		 return f[gname].attrs[aname]


def Contains(f, gname, cname):
	if (use_tables):
		if gname=="":
			return f.root.__contains__(cname)
		else:
			return f.root._f_get_child(gname).__contains__(cname)  #old: _f_getChild() 
	else:
		if gname=="":
			return f.__contains__(cname)
		else:
			return f[gname].__contains__(cname)


def ContainsGroup(group, cname):
	return group.__contains__(cname)


def CreateArray(f, where, aname, aval):
	if (use_tables):
		f.create_array(where, aname, aval)  #old createArray
	else:
		where.create_dataset(aname, data=aval)


def CreateGroup(f, gname):
	if (use_tables):
		return f.create_group(f.root, gname)
	else:
		return f.create_group(gname)


def SetAttr(where, aname, aval):
	if (use_tables):
		setattr(where._v_attrs, aname, aval)
	else:
		where.attrs.create(aname, aval)
