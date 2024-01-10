import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class WindTurbine:
    """
    A class to model a wind turbine's power output characteristics.

    Attributes:
    blade_radius (float): Blade radius in meters [m].
    performance_coefficient (float): Performance coefficient, no unit [-].
    air_density (float): Air density in kilograms per cubic meter [kg/m^3].
    nominal_power (float): Nominal power output in kilowatts [kW].
    nominal_wind_speed (float): Wind speed at which turbine reaches nominal power [m/s].
    cut_out_speed (float): Wind speed at which turbine shuts down to prevent damage [m/s].
    cut_in_speed (float): Minimum wind speed required for power generation [m/s].
    """

    def __init__(self, blade_radius, performance_coefficient, air_density,
                 nominal_power, nominal_wind_speed, cut_out_speed, cut_in_speed, price=None, lifetime=None, height=None):
        self.blade_radius = blade_radius
        self.performance_coefficient = performance_coefficient
        self.air_density = air_density
        self.nominal_power = nominal_power
        self.nominal_wind_speed = nominal_wind_speed
        self.cut_out_speed = cut_out_speed
        self.cut_in_speed = cut_in_speed
        self.price = price  # in EUR
        self.lifetime = lifetime  # in years
        self.height = height  # in meters

    def get_power_simple(self, wind_speeds):
        """
        Calculate the power output profile of the wind turbine using a simplified version of the
        actual power curve.

        Parameters:
        wind_speeds (numpy.array): An array of wind speeds in meters per second [m/s].

        Returns:
        numpy.array: An array of power outputs in kilowatts [kW].
        """
        # Using piecewise to define the power output for different wind speed segments
        power_output = np.piecewise(wind_speeds,
                                    [wind_speeds < self.cut_in_speed,
                                     (wind_speeds >= self.cut_in_speed) & (
                                         wind_speeds <= self.nominal_wind_speed),
                                        (wind_speeds > self.nominal_wind_speed) & (
                                         wind_speeds < self.cut_out_speed),
                                        wind_speeds >= self.cut_out_speed],
                                    [0,
                                     lambda x: (x / self.nominal_wind_speed)**3 *
                                        self.nominal_power,
                                        self.nominal_power,
                                        0])
        return power_output

    def get_power_physical(self, wind_speeds):
        """
        Calculate the power output profile of the wind turbine using physical properties.

        Parameters:
        wind_speeds (numpy.array): An array of wind speeds in meters per second [m/s].

        Returns:
        numpy.array: An array of power outputs in kilowatts [kW].
        """
        # Calculate the area of the circle the blades cover
        area = np.pi * self.blade_radius**2

        # Calculate the power using the formula:
        # P = 0.5 * cp * rho * A * V^3
        # Where P is the power in watts, cp is the performance coefficient,
        # rho is the air density, A is the swept area of the blades, and
        # V is the wind speed in m/s.
        power_output = 0.5 * self.performance_coefficient * \
            self.air_density * area * wind_speeds**3

        # Convert watts to kilowatts
        power_output /= 1000

        # Apply cut-in and cut-out speeds
        power_output = np.where(
            (wind_speeds >= self.cut_in_speed) & (
                wind_speeds < self.cut_out_speed),
            power_output,
            0
        )

        # Limit power to nominal power
        power_output = np.where(
            power_output > self.nominal_power,
            self.nominal_power,
            power_output
        )

        return power_output

    def get_capacity_factor(self, wind_speeds, years, simple=True):
        """
        Calculate the capacity factor of the wind turbine. Defaults to using the simple

        Parameters:
        wind_speeds (numpy.array): An array of wind speeds in meters per second [m/s].
        years (int): The number of years to calculate the capacity factor over.

        Returns:
        float: The capacity factor as a percentage [%].
        """
        # Calculate the power output profile in kilowatts [kW]
        if simple:
            power_output = self.get_power_simple(wind_speeds)
        else:
            power_output = self.get_power_physical(wind_speeds)

        # Calculate the number of hours in a year
        hours_per_year = 365 * 24

        # Wind speed is sampled hourly, so each element in power_output represents kilowatt-hours [kWh]
        # Calculate the capacity factor as a percentage [%]
        capacity_factor = (power_output.sum() /
                           (self.nominal_power * years * hours_per_year)) * 100

        return capacity_factor

    def plot_simple_power_curve(self):
        """
        Plot the power output vs wind speed curve using the simplified method.
        Parameters: None
        Returns:
        None
        """
        # Generate wind speeds from 0 to a bit beyond the cut-out speed
        wind_speeds = np.linspace(0, self.cut_out_speed + 5, num=1000)
        # Initialize the power output array
        power_output_to_plot = self.get_power_simple(wind_speeds)

        # Plot the power curve
        plt.figure(figsize=(16, 10))
        plt.plot(wind_speeds, power_output_to_plot,
                 label='Wind Turbine Power Curve')
        plt.axhline(y=self.nominal_power, color='r',
                    linestyle='--', alpha=0.33, label='Nominal Power')
        plt.axvline(x=self.cut_in_speed, color='g',
                    linestyle='--', label='Cut-in Speed')

        plt.axvline(x=self.nominal_wind_speed, color='orange',
                    linestyle='--', label='Nominal Wind Speed')
        plt.axvline(x=self.cut_out_speed, color='b',
                    linestyle='--', label='Cut-out Speed')
        plt.title('Wind Turbine Power Output (Simple cubic) vs. Wind Speed')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Power Output (kW)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_physical_power_curve(self):
        """
        Plot the power output vs wind speed curve using the physical method.
        Parameters: None
        Returns:
        None
        """
        # Generate wind speeds from 0 to a bit beyond the cut-out speed
        wind_speeds = np.linspace(0, self.cut_out_speed + 5, num=1000)
        # Initialize the power output array
        power_output_to_plot = self.get_power_physical(wind_speeds)

        # Plot the power curve
        plt.figure(figsize=(16, 10))
        plt.plot(wind_speeds, power_output_to_plot,
                 label='Wind Turbine Power Curve')
        plt.axhline(y=self.nominal_power, color='r',
                    linestyle='--', alpha=0.33, label='Nominal Power')
        plt.axvline(x=self.cut_in_speed, color='g',
                    linestyle='--', label='Cut-in Speed')

        plt.axvline(x=self.nominal_wind_speed, color='orange',
                    linestyle='--', label='Nominal Wind Speed')
        plt.axvline(x=self.cut_out_speed, color='b',
                    linestyle='--', label='Cut-out Speed')
        plt.title('Wind Turbine Power Output (Physical) vs. Wind Speed')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Power Output (kW)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_annual_energy_production(self, wind_speeds, years, simple=True):
        """
        Calculate the total average annual energy production from data of possibly multiple years. 

        Parameters:
        wind_speeds (numpy.array): An array of hourly wind speeds in meters per second [m/s].
        years (int): Period of data

        Returns:
        float: The annual energy production in kilowatt-hours [kWh].
        """
        # Calculate the power output profile in kilowatts [kW]
        if simple:
            power_output = self.get_power_simple(wind_speeds)
        else:
            power_output = self.get_power_physical(wind_speeds)

        # Power output is in kilowatts [kW] and it is hourly, therefore the same as kilowatt-hours [kWh]

        total_energy_production = power_output.sum()
        average_annual_total_energy_production = total_energy_production / years
        return average_annual_total_energy_production


class WindFarm:
    """
    A class that contains multiple wind turbines.
    Attributes:
    turbines (list): A list of WindTurbine objects.
    """

    def __init__(self, turbines):
        self.turbines = turbines

    def get_total_price(self) -> int:
        """
        Returns the total price of the wind farm in EUR.
        """
        total_price = 0
        for turbine in self.turbines:
            total_price += turbine.price
        return total_price

    def get_power_output_simple(self, wind_speeds):
        """
        Calculate the total power output profile of the wind farm using a simple cubic formula.
        Parameters:
        wind_speeds (numpy.array): An array of wind speeds in meters per second [m/s].
        Returns:
        numpy.array: An array of power outputs in kilowatts [kW].
        """
        # wind_speeds is a DataFrame, convert to numpy array
        if type(wind_speeds) == pd.core.frame.DataFrame:
            wind_speeds = wind_speeds.to_numpy()
        power_output = np.zeros(len(wind_speeds))
        for turbine in self.turbines:
            power_output += turbine.get_power_simple(wind_speeds)
        return power_output

    def get_power_output_physical(self, wind_speeds):
        """
        Calculate the total power output profile of the wind farm using physical properties.
        Parameters:
        wind_speeds (numpy.array): An array of wind speeds in meters per second [m/s].
        Returns:
        numpy.array: An array of power outputs in kilowatts [kW].
        """
        # wind_speeds is a DataFrame, convert to numpy array
        if type(wind_speeds) == pd.core.frame.DataFrame:
            wind_speeds = wind_speeds.to_numpy()
        power_output = np.zeros(len(wind_speeds))
        for turbine in self.turbines:
            power_output += turbine.get_power_physical(wind_speeds)
        return power_output

    def get_capacity_factor(self, wind_speeds, years, simple=True):
        """
        Calculate the capacity factor of the wind farm.
        Parameters:
        wind_speeds (numpy.array): An array of wind speeds in meters per second [m/s].
        years (int): The number of years to calculate the capacity factor over.
        Returns:
        float: The capacity factor as a percentage [%].
        """
        if simple:
            power_output = self.get_power_output_simple(wind_speeds)
        else:
            power_output = self.get_power_output_physical(wind_speeds)

        # Wind speed is sampled hourly, so each element in power_output represents kilowatt-hours [kWh]
        wind_farm_energy_produced = power_output.sum()
        wind_farm_total_nominal_power = np.sum(
            [turbine.nominal_power for turbine in self.turbines])

        # Calculate the number of hours in a year
        hours_per_year = 365 * 24

        # Calculate the capacity factor as a percentage [%]
        capacity_factor = (wind_farm_energy_produced /
                           (wind_farm_total_nominal_power * years * hours_per_year)) * 100
        return capacity_factor

    def get_annual_energy_production(self, wind_speeds, years, simple=True):
        """
        Calculate the total average annual energy production from data of possibly multiple years. 

        Parameters:
        wind_speeds (numpy.array): An array of hourly wind speeds in meters per second [m/s].
        years (int): Period of data

        Returns:
        float: The annual energy production in kilowatt-hours [kWh].
        """
        if simple:
            power_output = self.get_power_output_simple(wind_speeds)
        else:
            power_output = self.get_power_output_physical(wind_speeds)

        # Power output is in kilowatts [kW] and it is hourly, therefore the same as kilowatt-hours [kWh]
        total_energy_production = power_output.sum()
        average_annual_total_energy_production = total_energy_production / years
        return average_annual_total_energy_production


class WindFarmBuilder:
    """
    Produces WindFarm objects.
    """

    def __init__(self):
        pass

    # This function takes a list of wind turbines and a list of quantities of each turbine
    def build_wind_farm(self, list_of_wind_turbines, list_of_quantities):
        """
        Build a wind farm from a list of wind turbines and a list of quantities of each turbine.
        Parameters:
        list_of_wind_turbines (list): A list of WindTurbine objects.
        list_of_quantities (list): A list of integers representing the number of each turbine in the wind farm.
        Returns:
        WindFarm: A WindFarm object.
        """
        # Initialize an empty list of turbines
        wind_farm_turbines = []

        # For each turbine type and quantity
        for turbine, quantity in zip(list_of_wind_turbines, list_of_quantities):
            # Add the turbine to the list of turbines the wind farm will have
            wind_farm_turbines += [turbine] * quantity

        # Create a wind farm from the list of turbines
        wind_farm = WindFarm(wind_farm_turbines)

        return wind_farm


class SolarFarm:
    """
    A class that will simular a solar farm.
    """

    def __init__(self, area, pricepersqm=None, efficiency=0.2) -> None:
        self.area = area  # in m^2
        self.pricepersqm = pricepersqm
        self.efficiency = efficiency

    def get_power_output(self, df_weather):
        """
        Calculate the total power output profile of the solar farm.
        Parameters:
        df_weather (pandas.DataFrame): A DataFrame of hourly weather data.
        Returns:
        numpy.array: An array of power outputs in kilowatts [kW].
        """
        # Calculate the power output profile in kilowatts [kW]
        power_output = self.area * \
            df_weather['irradiance_total'].to_numpy() * self.efficiency
        return power_output

    def get_annual_energy_output(self, df_weather, years):
        """
        Calculate the total average annual energy production from data of possibly multiple years. 

        Parameters:
        df_weather (pandas.DataFrame): A DataFrame of hourly weather data.
        years (int): Period of data

        Returns:
        float: The annual energy production in kilowatt-hours [kWh].
        """
        power_output = self.get_power_output(df_weather)
        total_energy_production = power_output.sum()
        average_annual_total_energy_production = total_energy_production / years
        return average_annual_total_energy_production


class MicroGrid:
    """
    A class with the purpose of simulating a microgrid.
    """

    def __init__(self, windfarm: WindFarm, solarfarm: SolarFarm, storage_capacity_kwh, storage_efficiency, storage_price_per_kw) -> None:
        self.windfarm = windfarm
        self.solarfarm = solarfarm
        self.storage_capacity_kwh = storage_capacity_kwh
        self.storage_efficiency = storage_efficiency
        self.storage_price_per_kw = storage_price_per_kw 

    def get_total_price(self) -> int:
        wind_price = self.windfarm.get_total_price()
        solar_price = self.solarfarm.pricepersqm * self.solarfarm.area
        storage_price = self.storage_price_per_kw * self.storage_capacity_kwh
        return wind_price + solar_price + storage_price

    def get_power_output(self, df_weather):
        """
        Calculate the total power output profile of the microgrid.
        Parameters:
        df_weather (pandas.DataFrame): A DataFrame of hourly weather data.
        Returns:
        numpy.array: An array of power outputs in kilowatts [kW].
        """
        # Calculate the power output profile in kilowatts [kW]
        power_output = self.windfarm.get_power_output_physical(
            df_weather['wind_speed'].to_numpy()) + self.solarfarm.get_power_output(df_weather)
        return power_output

    def get_net_power(self, df_hourlyload_1year, df_weather_hourly_1year, plot=False):
        """
        Calculates the net power output of a wind farm over a year.
        Parameters:
        df_hourlyload_1year (pandas.DataFrame): A DataFrame of hourly energy consumption in kW.
        df_weather_hourly_1year (pandas.DataFrame): A DataFrame of hourly weather data.
        windfarm (WindFarm): A WindFarm object.
        plot (bool): Whether to plot the power output over time.
        Returns:
        pandas.DataFrame: A DataFrame of net power output in kW.
        """
        # Calculate the power output of the wind farm
        power_output = self.windfarm.get_power_output_physical(
            df_weather_hourly_1year['wind_speed'].to_numpy()) + self.solarfarm.get_power_output(df_weather_hourly_1year)

        net_power = df_hourlyload_1year.copy()
        # Drop the load column
        net_power = net_power.drop(columns=['Load (Watt)'])
        # Add the power output column
        net_power['Power Output (kW)'] = power_output
        # Calculate the net power
        net_power['Net Power (kW)'] = net_power['Power Output (kW)'] - \
            net_power['Load (kW)']

        if plot:
            # Plot the net power
            plt.figure(figsize=(16, 10))
            plt.plot(net_power.index, net_power['Net Power (kW)'])
            plt.title('Net Power Output Over Time')
            # Add 0 x-axis line
            plt.axhline(y=0, color='r', linestyle='--', alpha=1)
            plt.xlabel('Timestamp')
            plt.ylabel('Power Output (kW)')
            plt.show()

        net_power['Net Power Storage (kW)'] = self.simulate_energy_storage(
            net_power)

        return net_power

    def simulate_energy_storage(self, df):
        """
        Simulate the addition of an energy storage system to a dataset of power generation and consumption.

        Parameters:
        df (DataFrame): A DataFrame with columns 'Power Output (kW)', 'Load (kW)', 'Net Power (kW)' representing hourly data.
        storage_capacity_kWh (float): The capacity of the energy storage system in kWh.

        Returns:
        np.array: An array containing the new net power after considering the battery storage.
        """
        # self.storage_efficiency is 0.8 by default
        battery_state = self.storage_capacity_kwh * 0.5  # Initial state of charge of the battery
        new_net_power = []

        for index, row in df.iterrows():
            net_energy = row['Net Power (kW)']  # Net power for the hour

            if net_energy > 0:  # Excess energy
                energy_to_charge = min(
                    net_energy, self.storage_capacity_kwh - battery_state)
                battery_state += energy_to_charge
                new_net_power.append(net_energy - energy_to_charge)

            elif net_energy < 0:  # Energy deficit
                energy_to_discharge = min(-net_energy * 1/self.storage_efficiency, battery_state)
                battery_state -= energy_to_discharge
                adjusted_net_energy = net_energy + energy_to_discharge * self.storage_efficiency
                new_net_power.append(
                    adjusted_net_energy if adjusted_net_energy < 0 else 0)

        return np.array(new_net_power)

class DieselGenerator:
    """
    A class that will simulate a diesel generator.
    properties:
    diesel_price (float): price of diesel in EUR per liter
    efficiency (float): efficiency of the diesel generator
    fuel_consumption (float): liter fuel per kwh
    price_per_rated_kw (float): price of the diesel generator in EUR per rated kW
    """

    def __init__(self, diesel_price, fuel_consumption, price_per_rated_kw) -> None:
        self.diesel_price = diesel_price
        self.fuel_consumption = fuel_consumption
        self.price_per_rated_kw = price_per_rated_kw

    def get_generator_specs(self, net_power: np.array):
        """
        Calculate the specs of the diesel generator.
        Parameters:
        df_net_power (pandas.DataFrame): A DataFrame of net power output in kW.
        Returns:
        float: The running cost in EUR.
        """
        # Required rated power is the absolute value of minimum net power
        required_rated_power = abs(net_power.min())
        return required_rated_power

    def get_running_cost(self, net_power: np.array):
        """
        Calculate the running cost of the diesel generator.
        Parameters:
        df_net_power (pandas.DataFrame): A DataFrame of net power output in kW.
        Returns:
        float: The running cost in EUR.
        """
        # Calculate the sum of the net power output where elements are < 0
        total_kwh_to_generate = abs(net_power[net_power < 0].sum())
        # Calculate the total liters of diesel required
        total_liters = total_kwh_to_generate * self.fuel_consumption
        # Calculate the running cost
        running_cost = total_liters * self.diesel_price
        return running_cost

    def get_total_price(self, net_power: np.array):
        """
        Calculate the total price of the diesel generator.
        Parameters:
        df_net_power (pandas.DataFrame): A DataFrame of net power output in kW.
        Returns:
        float: The total price in EUR.
        """
        # Calculate the required rated power
        required_rated_power = self.get_generator_specs(net_power)
        # Calculate the running cost
        running_cost = self.get_running_cost(net_power)
        # Calculate the total price
        total_price = required_rated_power * self.price_per_rated_kw + running_cost
        return total_price
