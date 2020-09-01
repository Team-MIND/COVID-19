import numpy as np
from scipy.integrate import odeint
# Adding exposed delay before infectious
# derived from: # Found a better repo for this here: 
# implement the SEIR model as an extenable wrapper class

class SEIRModel:
	def __init__(self):
		self.population = -1
		self.days_infective = -1.0
		self.r0_base = -1.0
		self.delta=-1
		self.lockdown=-1
		self.low = -1.0

		self.susceptible = self.population - 1
		self.infected = 1
		self.recovered = 0
		self.dead = 0

		self.recov_ratio = -1
		self.incubation = -1
		self.exposed = 0
		self.death_chance = 0.1 #modify to be demo based.
		self.death_time = 1 / 12
		self.time_domain = None

	def reset(self):
		self.susceptible = self.population - 1
		self.infected = 1
		self.recovered = 0
		self.exposed = 0

	def set_vars(self,pop, Dur, r0, r0_info, incub):
		self.population = pop
		self.days_infective = Dur
		self.r0_base = r0 
		self.delta, self.lockdown, self.low =r0_info

		self.recov_ratio = 1.0 / self.days_infective
		self.incubation = incub

		self.susceptible = self.population - 1

	def set_incubation_period(self, value):
		self.incubation = value

	def set_death_chance(self): #made to then accept ages etc and calcuated it 
		return .1
	
	def set_r0_vars(self,base,low,delta,infelction):
		self.r0_base = base
		self.low =low
		self.delta=delta
		self.lockdown= infelction
	
	#needs to be modded for better fit...
	def infections(self, t):
		return self.daily_r0(t) * self.recov_ratio
	
	def daily_r0(self,t):
		return (self.r0_base-self.low) / (1 + np.exp(-self.delta*(self.lockdown - t))) + self.low #*np.exp(-.1*(t - self.lockdown))
	
	# The SEIR model differential equations.
	def deriv_seir(self, y, t):
		S, E, I, R, D = y
		infections = self.infections(t) * S * I / self.population
		symptomatic = E / self.incubation
		recoveries = (1- self.death_chance) * self.recov_ratio * I
		deaths = self.death_chance * self.death_time * I
		dSdt = -infections
		dEdt = infections - symptomatic
		dIdt = symptomatic - recoveries -deaths
		dRdt = recoveries
		dDdt = deaths
		return dSdt, dEdt, dIdt, dRdt, dDdt

	def run_period(self, days):
		t = np.linspace(0, days, days)
		S0, E0, I0, R0, D0 = (self.population-1), 0, 1, 0, 0 
		y0 = [S0, E0, I0, R0, D0] # Initial conditions vector
		
		# Integrate the SIR equations over the time grid, t.
		results = odeint(self.deriv_seir, y0, t)
		return results.T


if __name__ == '__main__':
	population= 1600000
	infection_duration= 14 #roughly two weeks
	r0_base= 3
	info= .008, 28, .7 #shetler in place was started 3/20
	incubation =5 # https://annals.org/aim/fullarticle/2762808/incubation-period-coronavirus-disease-2019-covid-19-from-publicly-reported

	model =SEIRModel()
	model.set_vars(population, infection_duration ,r0_base , info, incubation)
	days= 65
	S, E, I, R, D = model.run_period(days)
	print(S)