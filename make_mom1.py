from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt


def fetch_data(filename):
    data, header = fits.getdata(filename, header=True)

    # Get HI velocity grid
    _channels = np.arange(1, header["NAXIS3"] + 1)
    velos = (_channels - header["CRPIX3"]) * header["CDELT3"] / 1.0e3  # in km/s

    # Build 2d header
    wcs = WCS(header).celestial
    header_2d = wcs.to_header()

    print(velos)
    print(data)
    print(header_2d)
    print(wcs)

    return data, header_2d, wcs, velos


def calc_moments(data, velos, cliplevel=0.1):
    """
    Calculates the first two moments of a 3D data cube.

    Parameters
    ----------
    data : np.ndarray
        Input 3D HI data, assumed to be brightness temperature in Kelvin
    velos : np.ndarray
        Radial velocity of the HI data, in km/s. Shape must match data.shape[0]
    cliplevel : float
        The parameter to tune in order to suppress artifacts from noise

    Returns
    -------
    mom0, mom1
        First two moments, units are K.km/s and km/s, respectively
    """

    d_clipped = np.where(data > cliplevel, data, 0.0)
    mom0 = d_clipped.sum(axis=0)
    mom1 = np.sum(velos[:, None, None] * d_clipped, axis=0) / mom0

    return mom0, mom1


def plot_mom1(mom1, header, outname=None):
    wcs = WCS(header)

    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={"projection": wcs.celestial})

    im = ax.imshow(mom1, origin="lower", interpolation="nearest")

    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label(r"$v_{\rm LSR}\ [\rm km/s]$")

    # Grid and labels
    ax.grid(color='black', linestyle='solid')
    ax.set_xlabel(r'$\rm R.A.\ [deg]$')
    ax.set_ylabel(r'$\rm Declination\ [deg]$')

    # Save to disk
    if outname is not None:
        plt.savefig(outname, dpi=300)

    return


if __name__ == "__main__":

    # Load data
    data, header_2d, wcs, velos = fetch_data("labh_glue.fits")

    # Calculate moment maps
    mom0, mom1 = calc_moments(data, velos, cliplevel=0.1)

    # Plot
    plot_mom1(mom1, header_2d, outname="mom1_test.pdf")

    # Save
    fits.writeto("mom1_test.fits", mom1, header_2d, overwrite=True)
