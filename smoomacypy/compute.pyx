# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
from libc.math cimport sin, cos, asin, sqrt, exp, pow as _pow
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np

DTYPE = np.double
ctypedef np.double_t DTYPE_t
ctypedef float (*SMOOTH_FUNC)(float, float, float) nogil

cdef struct PtValueGeo:
    float x
    float y
    float cos_y
    float value

cdef struct PtValue:
    float x
    float y
    float value

cdef struct Pt2ValueGeo:
    float x
    float y
    float cos_y
    float value1
    float value2

cdef struct Pt2Value:
    float x
    float y
    float value1
    float value2


cdef inline float pareto(float alpha, float beta, float dist) nogil:
    return _pow((1.0 + alpha * dist), -beta)


cdef inline float exponential(float alpha, float beta, float dist) nogil:
    return exp(-alpha * _pow(dist, beta))


cdef inline float haversine2(float lon1, float lat1, float coslat1, float lon2, float lat2, float coslat2) nogil:
    cdef float dlon = lon2 - lon1
    cdef float dlat = lat2 - lat1
    cdef float a = sin(dlat/2)**2 + coslat1 * coslat2 * sin(dlon/2)**2

    return 12742000 * asin(sqrt(a))

cdef inline float euclidian(float x1, float y1, float x2, float y2) nogil:
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
        if lonlat:
            return compute_1_var_geo(knownpts, XI, YI, span, alpha, beta, expfunc)
        else:
            return compute_1_var_euclidian(knownpts, XI, YI, span, alpha, beta, expfunc)
    else:
        if lonlat:
            return compute_2_var_geo(knownpts, XI, YI, span, alpha, beta, expfunc)
        else:
            return compute_2_var_euclidian(knownpts, XI, YI, span, alpha, beta, expfunc)


cdef compute_1_var_geo(
        np.double_t[:,::1] knownpts, np.double_t[::1] XI, np.double_t[::1] YI,
        double span, double alpha, double beta, bint expfunc):
    cdef SMOOTH_FUNC smooth
    cdef Py_ssize_t len_xi = <Py_ssize_t>XI.shape[0]
    cdef Py_ssize_t len_yi = <Py_ssize_t>YI.shape[0]
    cdef Py_ssize_t nb_pot = <Py_ssize_t>len_xi * len_yi
    cdef Py_ssize_t nb_pts = knownpts.shape[0]
    cdef Py_ssize_t ix_x
    cdef Py_ssize_t ix_y
    cdef DTYPE_t x_cell
    cdef DTYPE_t y_cell
    cdef DTYPE_t cos_y_cell
    cdef Py_ssize_t j
    cdef DTYPE_t _sum = 0.0
    cdef Py_ssize_t ix = 0
    cdef DTYPE_t dist
    cdef np.double_t[:] point
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] res = np.zeros(nb_pot, dtype=DTYPE)
    cdef PtValueGeo *c_knownpts = <PtValueGeo *>malloc(nb_pts * sizeof(PtValueGeo))
    cdef PtValueGeo *ptval
    if not c_knownpts:
        raise MemoryError()

    with nogil:
        for j in range(nb_pts):
            point = knownpts[j]
            c_knownpts[j].x = point[<Py_ssize_t>0]
            c_knownpts[j].y = point[<Py_ssize_t>1]
            c_knownpts[j].cos_y = cos(point[<Py_ssize_t>1])
            c_knownpts[j].value = point[<Py_ssize_t>2]

        if expfunc:
            smooth = exponential
        else:
            smooth = pareto
    
        for ix_x in range(len_xi):
            x_cell = XI[ix_x]
            for ix_y in range(len_yi):
                y_cell= YI[ix_y]
                cos_y_cell = cos(y_cell)
                _sum = 0.0
                for j in range(nb_pts):
                    ptval = &c_knownpts[j]
                    dist = haversine2(x_cell, y_cell, cos_y_cell, ptval.x, ptval.y, ptval.cos_y)
                    _sum += ptval.value * smooth(alpha, beta, dist)
                res[ix] = _sum
                ix += 1
        free(c_knownpts)
    return res


cdef compute_1_var_euclidian(
        np.double_t[:,::1] knownpts, np.double_t[::1] XI, np.double_t[::1] YI,
        double span, double alpha, double beta, bint expfunc):
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
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] res = np.zeros(nb_pot, dtype=DTYPE)
    cdef PtValue *c_knownpts = <PtValue *>malloc(nb_pts * sizeof(PtValue))
    if not c_knownpts:
        raise MemoryError()

    with nogil:
        for j in range(nb_pts):
            point = knownpts[j]
            c_knownpts[j].x = point[<Py_ssize_t>0]
            c_knownpts[j].y = point[<Py_ssize_t>1]
            c_knownpts[j].value = point[<Py_ssize_t>2]

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
                    dist = euclidian(x_cell, y_cell, c_knownpts[j].x, c_knownpts[j].y)
                    _sum += c_knownpts[j].value * smooth(alpha, beta, dist)
                res[ix] = _sum
                ix += 1
        free(c_knownpts)
    return res

cdef compute_2_var_geo(
        np.double_t[:,::1] knownpts, np.double_t[::1] XI, np.double_t[::1] YI,
        double span, double alpha, double beta, bint expfunc):
    cdef SMOOTH_FUNC smooth
    cdef Py_ssize_t len_xi = <Py_ssize_t>XI.shape[0]
    cdef Py_ssize_t len_yi = <Py_ssize_t>YI.shape[0]
    cdef Py_ssize_t nb_pot = <Py_ssize_t>len_xi * len_yi
    cdef Py_ssize_t nb_pts = knownpts.shape[0]
    cdef Py_ssize_t ix_x
    cdef Py_ssize_t ix_y
    cdef DTYPE_t x_cell
    cdef DTYPE_t y_cell
    cdef DTYPE_t cos_y_cell
    cdef Py_ssize_t j
    cdef DTYPE_t _sum1 = 0.0
    cdef DTYPE_t _sum2 = 0.0
    cdef DTYPE_t t = 0.0
    cdef Py_ssize_t ix = 0
    cdef DTYPE_t dist
    cdef np.double_t[:] point
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] res = np.zeros(nb_pot, dtype=DTYPE)
    cdef Pt2ValueGeo *c_knownpts = <Pt2ValueGeo *>malloc(nb_pts * sizeof(Pt2ValueGeo))
    if not c_knownpts:
        raise MemoryError()

    with nogil:
        for j in range(nb_pts):
            point = knownpts[j]
            c_knownpts[j].x = point[<Py_ssize_t>0]
            c_knownpts[j].y = point[<Py_ssize_t>1]
            c_knownpts[j].cos_y = cos(point[<Py_ssize_t>1])
            c_knownpts[j].value1 = point[<Py_ssize_t>2]
            c_knownpts[j].value2 = point[<Py_ssize_t>3]

        if expfunc:
            smooth = exponential
        else:
            smooth = pareto
    
        for ix_x in range(len_xi):
            x_cell = XI[ix_x]
            for ix_y in range(len_yi):
                y_cell= YI[ix_y]
                cos_y_cell = cos(y_cell)
                _sum1 = 0.0
                _sum2 = 0.0
                for j in range(nb_pts):
                    dist = haversine2(x_cell, y_cell, cos_y_cell, c_knownpts[j].x, c_knownpts[j].y, c_knownpts[j].cos_y)
                    t = smooth(alpha, beta, dist)
                    _sum1 += c_knownpts[j].value1 * t
                    _sum2 += c_knownpts[j].value2 * t
                res[ix] = _sum1 / _sum2
                ix += 1
        free(c_knownpts)
    return res

cdef compute_2_var_euclidian(
        np.double_t[:,::1] knownpts, np.double_t[::1] XI, np.double_t[::1] YI,
        double span, double alpha, double beta, bint expfunc):
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
    cdef DTYPE_t _sum1 = 0.0
    cdef DTYPE_t _sum2 = 0.0
    cdef DTYPE_t t = 0.0
    cdef Py_ssize_t ix = 0
    cdef DTYPE_t dist
    cdef np.double_t[:] point
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] res = np.zeros(nb_pot, dtype=DTYPE)
    cdef Pt2Value *c_knownpts = <Pt2Value *>malloc(nb_pts * sizeof(Pt2Value))
    if not c_knownpts:
        raise MemoryError()

    with nogil:
        for j in range(nb_pts):
            point = knownpts[j]
            c_knownpts[j].x = point[<Py_ssize_t>0]
            c_knownpts[j].y = point[<Py_ssize_t>1]
            c_knownpts[j].value1 = point[<Py_ssize_t>2]
            c_knownpts[j].value2 = point[<Py_ssize_t>3]

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
                    dist = euclidian(x_cell, y_cell, c_knownpts[j].x, c_knownpts[j].y)
                    t = smooth(alpha, beta, dist)
                    _sum1 += c_knownpts[j].value1 * t
                    _sum2 += c_knownpts[j].value2 * t
                res[ix] = _sum1 / _sum2
                ix += 1
        free(c_knownpts)
    return res

