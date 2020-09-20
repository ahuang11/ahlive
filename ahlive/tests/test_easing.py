import numpy as np
import easing_functions


def test_interp():
    for method in dir(easing_functions):
        try:
            ease = getattr(easing_functions, method)(start=0, end=1)
        except:
            continue
        x = np.arange(0, 1, 0.1)
        expected = np.array(map(ease.ease, x))

        if 'InOut' in method:
            how = 'in_out'
        elif 'In' in method:
            how = 'in'
        elif 'Out' in method:
            how = 'out'
        method = (
            method.replace('Ease', '').replace('In', '').replace('Out', ''))
        easing = Easing(method=method, how=how, boomerang=False)
        actual = easing.interp(np.array([0, 1]))

        np.testing.assert_almost_equal(expected, actual)
