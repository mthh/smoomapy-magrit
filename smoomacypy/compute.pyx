# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
from libc.math cimport sin, cos, asin, sqrt, exp, pow as _pow
import numpy as np
cimport numpy as np


DTYPE = np.double
ctypedef np.double_t DTYPE_t
ctypedef float (*DIST_FUNC)(float, float, float, float)
ctypedef float (*SMOOTH_FUNC)(float, float, float)


cdef inline float pareto(float alpha, float beta, float dist) nogil:
    return _pow((1.0 + alpha * dist), -beta)


cdef inline float exponential(float alpha, float beta, float dist) nogil:
    return exp(-alpha * _pow(dist, beta))


cdef float haversine(float lon1, float lat1, float lon2, float lat2) nogil:
    cdef float dlon = lon2 - lon1
    cdef float dlat = lat2 - lat1
    cdef float a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    return 6371000 * 2 * asin(sqrt(a))


cdef float euclidian(float x1, float y1, float x2, float y2) nogil:
    cdef float a = x1 - x2
    cdef float b = y1 - y2
    return sqrt(a * a + b * b)


def _compute_stewart(knownpts, XI, YI, nb_var, type_function, span, beta, lonlat):
    if type_function == 'exponential':
        alpha = 0.6931471805 / pow(span, beta)
        expfunc = True
    else:
        alpha = (pow(2, 1/beta) - 1) / span
        expfunc = False

    if nb_var == 1:
        return compute_1_var(knownpts, XI, YI, span, alpha, beta, lonlat, expfunc)
    else:
        return compute_2_var(knownpts, XI, YI, span, alpha, beta, lonlat, expfunc)


cdef compute_1_var(
        np.double_t[:,::1] knownpts, np.double_t[::1] XI, np.double_t[::1] YI,
        double span, double alpha, double beta, bint lonlat, bint expfunc):
    cdef DIST_FUNC dist_func
    cdef SMOOTH_FUNC smooth
    cdef Py_ssize_t len_xi = <Py_ssize_t>XI.shape[0]
    cdef Py_ssize_t len_yi = <Py_ssize_t>YI.shape[0]
    cdef Py_ssize_t nb_pot = <Py_ssize_t>len_xi * len_yi
    cdef Py_ssize_t nb_pts = knownpts.shape[0]
    cdef Py_ssize_t ix_x
    cdef Py_ssize_t ix_y
    cdef DTYPE_t x_cell
    cdef DTYPE_t y_cell
    cdef Py_ssize_t j
    cdef DTYPE_t _sum = 0.0
    cdef Py_ssize_t ix = 0
    cdef DTYPE_t dist
    cdef np.double_t[:] point
    cdef np.ndarray res = np.zeros(nb_pot, dtype=DTYPE)

    if lonlat:
        dist_func = haversine
    else:
        dist_func = euclidian

    if expfunc:
        smooth = exponential
    else:
        smooth = pareto

    for ix_x in range(len_xi):
        x_cell = XI[ix_x]
        for ix_y in range(len_yi):
            y_cell= YI[ix_y]
            _sum = 0.0
            for j in range(nb_pts):
                point = knownpts[j]
                dist = dist_func(x_cell, y_cell, point[<Py_ssize_t>0], point[<Py_ssize_t>1])
                _sum += point[<Py_ssize_t>2] * smooth(alpha, beta, dist)
            res[ix] = _sum
            ix += 1
    return res


cdef compute_2_var(
        np.double_t[:,::1] knownpts, np.double_t[::1] XI, np.double_t[::1] YI,
        double span, double alpha, double beta, bint lonlat, bint expfunc):
    cdef DIST_FUNC dist_func
    cdef SMOOTH_FUNC smooth
    cdef Py_ssize_t len_xi = <Py_ssize_t>XI.shape[0]
    cdef Py_ssize_t len_yi = <Py_ssize_t>YI.shape[0]
    cdef Py_ssize_t nb_pot = <Py_ssize_t>len_xi * len_yi
    cdef Py_ssize_t nb_pts = knownpts.shape[0]
    cdef Py_ssize_t ix_x
    cdef Py_ssize_t ix_y
    cdef DTYPE_t x_cell
    cdef DTYPE_t y_cell
    cdef DTYPE_t _sum1 = 0.0
    cdef DTYPE_t _sum2 = 0.0
    cdef DTYPE_t t = 0.0
    cdef Py_ssize_t ix = 0
    cdef DTYPE_t dist
    cdef np.double_t[:] point
    cdef np.ndarray res = np.zeros(nb_pot, dtype=DTYPE)

    if lonlat:
        dist_func = haversine
    else:
        dist_func = euclidian

    if expfunc:
        smooth = exponential
    else:
        smooth = pareto

    for ix_x in range(len_xi):
        x_cell = XI[ix_x]
        for ix_y in range(len_yi):
            y_cell= YI[ix_y]
            _sum1 = 0.0
            _sum2 = 0.0
            for j in range(nb_pts):
                point = knownpts[j]
                dist = dist_func(x_cell, y_cell, point[<Py_ssize_t>0], point[<Py_ssize_t>1])
                t = smooth(alpha, beta, dist)
                _sum1 += point[<Py_ssize_t>2] * t
                _sum2 += point[<Py_ssize_t>3] * t
            res[ix] = np.true_divide(_sum1, _sum2)  # _sum1 / _sum2
            ix += 1
    return res
