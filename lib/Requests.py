import numpy as np
import pandas as pd
from lib.Constants import DIST_MAT, CONSTANT_SPEED, my_dist_class


class Req:
    """
    Req is a class for requests
    Attributes:
        id: sequential unique id
        Tr: request time
        ozone: origin zone
        dzone: destination zone
        Ds: shortest travel distance
        Ts: shortest travel time
        Tp: pickup time
        Td: dropoff time
        DR: distance rejected (true if distance O->D is less than DISTANCE_THRESHOLD)
    """

    def __init__(self, id, Tr, fare, ozone=4, dzone=12, DIST_MAT=DIST_MAT):
        """
        Creates a request instance.

        @param id: (int) sequential unique id
        @param Tr: (int) req time
        @param fare: (float)
        @param ozone: (int) origin zone
        @param dzone: (int) destination zone
        @param DIST_MAT: deprecated
        """
        self.id = id
        self.Tr = Tr
        self.ozone = ozone
        self.dzone = dzone
        # self.DIST_MAT = DIST_MAT
        self.Ds, self.Ts = self._get_distance_time()
        self.fare = fare

        self.Tp = -1.0
        self.Td = -1.0
        self.D = 0.0
        self.DR = False
        # self.NS = 0
        # self.NP = 0
        # self.ND = 0

    # @profile
    def _get_distance_time(self):
        """
        Gets the distance and time in the request.
        @return: tuple (distance, time)
        """
        try:

            ds = my_dist_class.return_distance(self.ozone, self.dzone)
            # ds = np.ceil(
            #     # self.DIST_MAT.query(
            #     #     "PULocationID == {origin} & DOLocationID == {destination} ".format(
            #     #         origin=self.ozone, destination=self.dzone
            #     #     )
            #     # )["trip_distance_meter"].values[0]
            #     # DIST_MAT[(DIST_MAT["PULocationID"] == self.ozone) & (DIST_MAT["DOLocationID"] == self.dzone)][
            #     #     "trip_distance_meter"].values[0]
            #     DIST_MAT.loc[self.ozone,  self.dzone]["trip_distance_meter"].values
            # )
        except:
            ds = 10 * 1609  # meter
            print("didn't find the distance")
        # Ts = np.int(ds / CONSTANT_SPEED)
        return ds, np.int(ds / CONSTANT_SPEED)

    def get_origin(self):
        """
        @return: (int) the origin zone id
        """
        return self.ozone

    def get_destination(self):
        """
        @return: (int) the destination zone id
        """
        return self.dzone

    # visualize
    # def draw(self):
    #     import matplotlib.pyplot as plt

    #     plt.plot(self.olng, self.olat, "r", marker="+")
    #     plt.plot(self.dlng, self.dlat, "r", marker="x")
    #     plt.plot(
    #         [self.olng, self.dlng],
    #         [self.olat, self.dlat],
    #         "r",
    #         linestyle="--",
    #         dashes=(0.5, 1.5),
    #     )

    def __str__(self):
        """
        Defines string representation of a request.
        @return: (str) "req [id] from [origin zone] to [dest. zone] at [time]"
        """
        str = "req %d from (%.7f) to (%.7f) at t = %.3f" % (
            self.id,
            self.ozone,
            self.dzone,
            self.Tr,
        )
        # str += "\n  earliest pickup time = %.3f, latest pickup at t = %.3f" % ( self.Cep, self.Clp)
        # str += "\n  pickup at t = %.3f, dropoff at t = %.3f" % ( self.Tp, self.Td)
        return str
