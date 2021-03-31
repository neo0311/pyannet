from pyannet.data_prep import *

def test_z_score_norm_1d():
    a = np.asarray((1,5,2,4,0,-3,-8.55))
    assert(data_transform(a,type='z_score_norm')[0]).all() == np.asarray(([ 0.21875728],[ 1.15390292],[ 0.45254369],[ 0.92011651],[-0.01502913],[-0.71638835],[-2.01390292])).all()
    

def test_min_max_norm_1d():
    a = np.asarray((1,5,2,4,0,-3,-8.55))
    assert(data_transform(a,type='min_max_norm'))[0].all() == np.asarray(([0.70479705],[1.        ],[0.77859779],[0.92619926],[0.63099631],[0.4095941 ],[0.        ])).all()

def test_min_max_norm_2d():
    a = np.asarray(([1,5,2],[4,0,-3,],[-3,5,6.8]))
    assert(data_transform(a,type='min_max_norm'))[0].all() == np.asarray(([0.57142857, 1.         ,0.51020408],[1.         ,0.         ,0.        ],[0.         ,1.         ,1.        ])).all()

def test_z_score_norm_2d():
    a = np.asarray(([1,5,2],[4,0,-3,],[-3,5,6.8]))
    assert(data_transform(a,type='z_score_norm'))[0].all() == np.asarray(([ 0.11624764,  0.70710678,  0.01666204],[ 1.16247639, -1.41421356, -1.23299088],[-1.27872403,  0.70710678,  1.21632884])).all()

def test_LHCSampling_samples_lie_within_sample_space():
    dimensionSpans = np.asarray(([1,69],[4,6]))
    A = LHCSampling(numSamples=50, numDimensions=2, numDivisions=80, dimensionSpans=dimensionSpans)
    assert(np.all((A[0,:]<69)))==True 
    assert(np.all((A[1,:]<6)))==True 
    assert(np.all((A[0,:]>1)))==True 
    assert(np.all((A[1,:]>4)))==True 

def test_QMCSampling_samples_lie_within_sample_space():
    dimensionSpans = np.asarray(([1,69],[4,6]))
    A = QMC_sampling(numSamples=60, numDimensions=2, dimensionSpans=dimensionSpans)
    assert(np.all((A[0,:]<69)))==True 
    assert(np.all((A[1,:]<6)))==True 
    assert(np.all((A[0,:]>1)))==True 
    assert(np.all((A[1,:]>4)))==True 