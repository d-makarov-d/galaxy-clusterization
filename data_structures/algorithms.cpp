#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSIONS
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <iostream>

class IndexComparator {
private:
    const double *data;
public:
    IndexComparator(const double *data){
        this->data = data;
    }

    bool operator()(const long &a, const long &b) const {
        double a_value = data[a];
        double b_value = data[b];
        return a_value == b_value ? a < b : a_value < b_value;
    }
};

/**
  * Uses nth_element function (https://en.cppreference.com/w/cpp/algorithm/nth_element) to partially sort given
  * array, using given split dimension
  * @param data: Pinter to data with actual points
  * @param idx_array: Pointer to index array, actually modified by nth_element
  * @param idx_start, idx_end: Indexes in idx_array, defining interval, in which nth_element works
  */
void nth_element_appl(const double *data, long *idx_array, const long idx_start, const long idx_end) {
    IndexComparator comparator(data);
    long *start = idx_array + idx_start;
    long *middle = idx_array + idx_start + (idx_end - idx_start) / 2;
    long *end = idx_array + idx_end;
    std::nth_element(start, middle, end, comparator);
}

/**
  * Decodes arguments, passed from python and calls nth_element_appl
  */
static PyObject *algorithm_nth_element(PyObject *self, PyObject *args) {
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    long idx_start, idx_end;

    if (!PyArg_ParseTuple(args, "OOll", &arg1, &arg2, &idx_start, &idx_end))
        return NULL;

    PyObject *arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr1 == NULL)
        return NULL;

    PyObject *arr2 = PyArray_FROM_OTF(arg2, NPY_LONG, NPY_IN_ARRAY);
    if (arr2 == NULL)
        return NULL;

    double *data = (double*) PyArray_DATA(arr1);
    long *idx_array = (long*) PyArray_DATA(arr2);

    nth_element_appl(data, idx_array, idx_start, idx_end);

    Py_DECREF(arr1);
    Py_DECREF(arr2);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef AlgorithmsMethods[] = {
    {"nts_element", algorithm_nth_element, METH_VARARGS, "nth element"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef algorithmsmodule = {
    PyModuleDef_HEAD_INIT,
    "algorithms",
    "Python interface for some 'algorithm' library functions",
    -1,
    AlgorithmsMethods
};

PyMODINIT_FUNC PyInit_algorithms(void) {
    import_array(); // SUKA SUKA SUKA
    return PyModule_Create(&algorithmsmodule);
}