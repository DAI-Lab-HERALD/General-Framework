import numpy as np
from scipy.interpolate import CubicSpline

from Adversarial_classes.helper import Helper


class Spline:
    @staticmethod
    def spline_data(X, Y, total_spline_values):
        """
        Process data with cubic spline interpolation after checking monotonicity and increasing order,
        ensuring x-values are increasing by adding small increments where necessary.

        Parameters:
        X, Y (numpy.ndarray): Input data arrays.
        total_spline_values (int): Number of spline values to generate.
        figure: Not used in this function but included for completeness.

        Returns:
        numpy.ndarray: Spline interpolated data.
        """
        # concatenate the data
        data = np.concatenate((X, Y), axis=-2)

        # check if data monotonic _> needed for spline function -> flip X,Y values if not
        monotonic_data = Helper.is_monotonic(data)
        data[~monotonic_data, :, :] = np.flip(
            data[~monotonic_data, :, :], axis=-1)

        # check if values are increasing -> needed for spline function -> flip trajectory if not
        increasing_data = Helper.is_increasing(data)
        data[~increasing_data, :, :] = np.flip(
            data[~increasing_data, :, :], axis=-2)

        # guarantee that the x values are increasing
        standing_still = Helper.compute_mask_values_standing_still(data)
        index_dim_2 = np.arange(data.shape[2])[:, np.newaxis]
        increment = index_dim_2 * 0.001
        increment = np.broadcast_to(increment, data.shape)

        data[standing_still] += increment[standing_still]

        # Cubic spline data
        spline_data = np.empty(
            (X.shape[0], X.shape[1], total_spline_values, X.shape[3]))

        # Spline all the data
        for i in range(spline_data.shape[0]):
            for j in range(spline_data.shape[1]):
                x = np.linspace(data[i, j, 0, 0],
                                data[i, j, -1, 0], total_spline_values)
                cs = CubicSpline(data[i, j, :, 0], data[i, j, :, 1])
                spline_data[i, j, :, 0] = x
                spline_data[i, j, :, 1] = cs(x)

        # Translate back to original data
        spline_data[~increasing_data, :, :] = np.flip(
            spline_data[~increasing_data, :, :], axis=-2)
        spline_data[~monotonic_data, :, :] = np.flip(
            spline_data[~monotonic_data, :, :], axis=-1)

        return spline_data
