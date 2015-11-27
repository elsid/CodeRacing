class PidController:
    def __init__(self, proportional_gain, integral_gain, derivative_gain):
        self.proportional_gain = proportional_gain
        self.integral_gain = integral_gain
        self.derivative_gain = derivative_gain
        self.__previous_output = 0
        self.__previous_error = 0
        self.__integral = 0

    def __call__(self, error):
        self.__integral += error
        derivative = error - self.__previous_error
        output = (self.proportional_gain * error +
                  self.integral_gain * self.__integral +
                  self.derivative_gain * derivative)
        self.__previous_output = output
        self.__previous_error = error
        return output
