class Time_Signature(object):
    def __init__(self, time_step=None, numerator=None, denominator=None, beat_division_factors=None):
        self.__time_step = None     # index corresponding to the time_step in hvo where signature change happens
        self.__numerator = None
        self.__denominator = None
        self.__beat_division_factors = None             # must be a list of integers

        if time_step is not None:
            self.time_step = time_step
        if numerator is not None:
            self.numerator = numerator
        if denominator is not None:
            self.denominator = denominator
        if beat_division_factors is not None:
            self.beat_division_factors = beat_division_factors

    def __repr__(self):
        rep = "Time_Signature = { \n " +\
              "\t time_step: {}, \n \t numerator: {}, \n \t denominator: {}, \n \t beat_division_factors: {}".format(
                  self.time_step, self.numerator, self.denominator, self.beat_division_factors)+"\n}"
        return rep

    def __eq__(self, other):
        # Make sure the types are the same
        assert isinstance(other, Time_Signature), "Expected a Time_Signature Instance but received {}".format(
            type(other))

        # ignore the start time index of time_signatures and check whether other fields are equal
        is_eq = all([
            self.numerator == other.numerator,
            self.denominator == other.denominator,
            self.beat_division_factors == other.beat_division_factors
        ])
        return is_eq

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def time_step(self):
        return self.__time_step

    @time_step.setter
    def time_step(self, val):
        if val is None:
            self.__time_step = None
        else:
            assert isinstance(val, int), "Time signature time should be an integer " \
                                         "corresponding to the time_step in HVO sequence"
            assert val >= 0, "Make sure the numerator is greater than or equal to zero"
            self.__time_step = val

    @property
    def numerator(self):
        return self.__numerator

    @numerator.setter
    def numerator(self, val):
        if val is None:
            self.__numerator = None
        else:  # Check consistency of input with the required
            assert isinstance(val, int), "Make sure the numerator is an integer"
            assert val > 0, "Make sure the numerator is greater than zero"
            # Now, safe to update the __time_signature local variable
            self.__numerator = val

    @property
    def denominator(self):
        return self.__denominator

    @denominator.setter
    def denominator(self, val):
        if val is None:
            self.__denominator = None
        else:  # Check consistency of input with the required
            # Ensure numerator is an integer
            assert isinstance(val, int), "Make sure the denominator is an integer"
            assert is_power_of_two(val), "Denominator must be binary (i.e. a value that is a power of 2)"
            assert val > 0, "Make sure the numerator is greater than zero"
            # Now, safe to update the __time_signature local variable
            self.__denominator = val

    @property
    def beat_division_factors(self):
        return self.__beat_division_factors

    @beat_division_factors.setter
    def beat_division_factors(self, x):
        if x is None:
            self.__denominator = None
        else:
            # Ensure input is a list
            assert isinstance(x, list), "Expected a list but received {}".format(type(x))
            # Ensure the values in list are integers
            assert all([isinstance(x_i, int) for x_i in x]), "Expected a list of int but received " \
                                                             "{}".format([type(x_i) for x_i in x])
            # Ensure the factors is list are either binary or multiple of 3
            assert all([((not is_power_of_two(factor)) or (factor % 3 != 0)) for factor in x]
                       ), "beat_division_factors must be either power of 2 or multiple of 3"
            # Now, Safe to update local beat_division_factors variable
            self.__beat_division_factors = x

    @property
    def is_ready_to_use(self):
        # Checks to see if all fields are filled and consequently, the Time_Signature is ready to be used externally
        fields_available = list()
        for key in self.__dict__.keys():
            fields_available.append(True) if self.__dict__[key] is not None else fields_available.append(False)
        return all(fields_available)


class Tempo(object):
    def __init__(self, time_step=None,  qpm=None):
        self.__time_step = None    # index corresponding to the time_step in hvo where signature change happens
        self.__qpm = None
        if time_step is not None:
            self.time_step = time_step
        if qpm is not None:
            self.qpm = qpm

    def __repr__(self):
        rep = "Tempo = { \n " +\
              "\t time_step: {}, \n \t qpm: {}".format(self.time_step, self.qpm)+"\n}"
        return rep

    @property
    def time_step(self):
        return self.__time_step

    @time_step.setter
    def time_step(self, val):
        if val is None:
            self.__time_step = None
        else:
            assert isinstance(val, int), "Starting time index for tempo should be an integer " \
                                         "corresponding to the time_step in HVO sequence. Received {}".format(type(val))
            assert val >= 0, "Make sure the numerator is greater than or equal to zero"
            self.__time_step = val

    @property
    def qpm(self):
        return self.__qpm

    @qpm.setter
    def qpm(self, val):
        if val is None:
            self.__qpm = None
        else:   # Check consistency of input with the required
            # Ensure numerator is an integer
            assert isinstance(val, (int, float)), "Make sure the qpm is a float"
            assert val > 0, "Make sure tempo is positive"
            # Now, safe to update the __qpm local variable
            self.__qpm = val

    @property
    def is_ready_to_use(self):
        # Checks to see if all fields are filled and consequently, the Time_Signature is ready to be used externally
        fields_available = list()
        for key in self.__dict__.keys():
            fields_available.append(True) if self.__dict__[key] is not None else fields_available.append(False)
        return all(fields_available)

    def __eq__(self, other):
        # Make sure the types are the same
        assert isinstance(other, Tempo), "Expected a Tempo Instance but received {}".format(
            type(other))

        # ignore the start time index of time_signatures and check whether qpms are equal
        return self.qpm == other.qpm

    def __ne__(self, other):
        return not self.__eq__(other)


def is_power_of_two(n):
    """
    Checks if a value is a power of two
    @param n:                               # value to check (must be int or float - otherwise assert error)
    @return:                                # True if a power of two, else false
    """
    if n is None:
        return False

    assert (isinstance(n, int) or isinstance(n, float)), "The value to check must be either int or float"

    if (isinstance(n, float) and n.is_integer()) or isinstance(n, int):
        # https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
        n = int(n)
        return (n & (n - 1) == 0) and n != 0
    else:
        return False
