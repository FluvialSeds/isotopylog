##############
# CO47 DICTS #
##############

# 1) dictionary holding all calibration equations, and corresponding constants

#passey and henkes constants (Eq. 5)
p4 = -3.407e9
p3 = 2.365e7
p2 = -2.607e3
p1 = -5.880
p0 = 0.280

#stolper and eiler constants (Fig. 3)
s4 = 1.006e9
s2 = 2.620e4
s0 = 0.2185

#bonifacie et al. constants (Eq. 2)
b4 = 1.0771e9
b2 = 2.5885e4
b0 = 0.1745

#Temp conversion constants
cdes_aff = 0.092 #25C - 90C
ghosh_aff = 0.081 #25C - 90C

#ref frame conversion constants
m = 1.0381 #Ghosh to CDES 25C
b = 0.0266 #Ghosh to CDES 25C

#make lambda equations for native (i.e., literature reported) equations

#PH12: CDES25 native
PH12 = lambda T : p4/(T**4) + p3/(T**3) + p2/(T**2) + p1/T + p0

#SE15: Ghosh25 native
SE15 = lambda T : s4/(T**4) + s2/(T**2) + s0

#Bea17: CDES90 native
Bea17 = lambda T : b4/(T**4) + b2/(T**2) + b0

#store in dictionary
caleqs = {'PH12':{'Ghosh25':lambda T : (PH12(T) - b)/m,
				  'Ghosh90': lambda T : (PH12(T) - b)/m - ghosh_aff,
				  'CDES25': lambda T : PH12(T),
				  'CDES90':lambda T : PH12(T) - cdes_aff},
		  'SE15':{'Ghosh25': lambda T : SE15(T),
		  		  'Ghosh90': lambda T : SE15(T) - ghosh_aff,
				  'CDES25':lambda T : SE15(T)*m + b,
				  'CDES90':lambda T : SE15(T)*m + b - cdes_aff},
		  'Bea17':{'Ghosh25':lambda T : (Bea17(T) - b)/m + ghosh_aff,
		  		  'Ghosh90' :lambda T : (Bea17(T) - b)/m,
				  'CDES25':lambda T : Bea17(T) + cdes_aff,
				  'CDES90': lambda T : Bea17(T)},
		 }


# 2) dictionary for holding "isotope parameters" (Daëron et al. 2016)

#in order: R13_VPDB, R18_VPDB, R17_VPDB, lam17
d47_isoparams = {'Gonfiantini':
					[0.0112372, #R13_VPDB
					 0.0020672, #R18_VPDB
					 0.00038592, #R17_VPDB
					 0.5164 #lam17
					],
				 'Brand':
					[0.01118, #R13_VPDB
					 0.0020672, #R18_VPDB
					 0.00039099, #R17_VPDB
					 0.528 #lam17
					],
				 'Craig + Assonov':
					[0.0112372, #R13_VPDB
					 0.0020672, #R18_VPDB
					 0.00039299, #R17_VPDB
					 0.528 #lam17
					],
				 'Chang + Li':
					[0.01118, #R13_VPDB
					 0.0020672, #R18_VPDB
					 0.00038413, #R17_VPDB
					 0.528 #lam17
					],
				 'Craig + Li':
					[0.0112372, #R13_VPDB
					 0.0020672, #R18_VPDB
					 0.00038606, #R17_VPDB
					 0.528 #lam17
					],
				 'Barkan':
					[0.01118, #R13_VPDB
					 0.00206774, #R18_VPDB
					 0.00039089, #R17_VPDB
					 0.528 #lam17
					],
				 'Passey':
					[0.01118, #R13_VPDB
					 0.00206774, #R18_VPDB
					 0.00039094, #R17_VPDB
					 0.528 #lam17
					],
				}


# 3) Dictionary for holding kDistribution model parameters for summary printing
mod_params = {'Hea14' : ['ln(kc)','ln(kd)','ln(k2)'],
			  'HH20' : ['ln(k_mu)','ln(k_sig)'],
			  'PH12' : ['ln(k)','intercept'],
			  'SE15' : ['ln(k1)','ln(k_dif_single)','[pair]_0/[pair]_eq']
			  }



# 4) Dictionary for holding clumped isotope names for summary printing
clump_isos = {'CO47' : ['D47','d13C_vpdb','d18O_vpdb'],
			 }

















