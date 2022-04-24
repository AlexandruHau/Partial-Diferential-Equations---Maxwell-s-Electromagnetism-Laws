# Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import scipy
import pandas as pd
import sys

# Generate function for the monopole charge distribution.
# This represents the initial conditions
def GenerateCharge(N, j):
	# Generate the monopole distribution. This is in accordance with the
	# Dirichlet Boundary Conditions
	distribution = np.zeros((N,N,N))
	
	# Assign value for the charge in the middle
	distribution[int(N/2)][int(N/2)][:] = j
	
	# Return the initial distribution
	return distribution
	
def ApplyBoundary(grid, charge):
	# Implement the boundary conditions
	N = len(grid[0][0])
	grid[0,:,:] = 0
	grid[(N-1),:,:] = 0
	grid[:,0,:] = 0
	grid[:,(N-1),:] = 0
	
	# Apply the condition of constant charge in
	# the middle wire as well
	# grid[int(N/2)][int(N/2)][:] = charge[int(N/2)][int(N/2)][:]

	# Return the overall grid	
	return grid
	

# Implement the Jacobi algorithm using the numpy function roll	
def JacobiMethod(grid, charge):
	# Update the overall grid using the roll function from
	# the numpy array
	grid = (1/6) * ( np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) + np.roll(grid, 1, axis=0)
	 + np.roll(grid, -1, axis=0) + np.roll(grid, 1, axis=2) + np.roll(grid, -1, axis=2) + charge)
	 
	# Apply the boundary conditions
	grid = ApplyBoundary(grid, charge)
	
	# Return the overall grid
	return grid
		
# Implement the Gauss-Seidel method without the numpy roll function	
def ManualGaussSeidel(grid, charge):
	# Now manually update the grid using the Gauss-Seidel algorithm. For this time,
	# we do not need the deep copy at each step. Take again the length of one row
	N = len(grid[0])
	for i in range(N):
		for j in range(N):
			for k in range(N):
			
				# Use the Gauss-Seidel method
				# Use the Jacobi method
				grid[i][j][k] = (1/6) * (grid[i][(j+1)%N][k] + grid[i][(j-1)%N][k] + grid[(i+1)%N][j][k] + grid[(i-1)%N][j]
				[k] + grid[i][j][(k+1)%N] + grid[i][j][(k-1)%N] + charge[i][j][k])
			
	# Apply again the boundary conditions
	grid = ApplyBoundary(grid, charge)
	
	# Return the final grid
	return grid
	
# Create function for animating the potential in the midplane cut in the
# cubic box. 
def AnimatePotential(N, j):
	# Initialize the charge grid
	potentialDistribution = GenerateCharge(N, j)
	
	# Make a deep copy of the charged grid to retain for any 
	# further operations
	charge = np.copy(potentialDistribution)
	
	# Now update the overall plot indefinitely in the while-loop
	# instruction. Use the count variable for control
	count = 0
	increment = 0
	while(count < 1):
		
		if(increment % 20 == 0):
			# Commands for animated plotting
			plt.cla()
			plt.imshow(potentialDistribution[int(N/2),:,:], animated=True)
			plt.draw()
			plt.pause(0.001)
		
		# Update the potential distribution
		potentialDistribution = JacobiMethod(potentialDistribution, charge)
		increment += 1
	
	
# Create function now for updating for more time steps the 
# charge distribution. Add a live plot at each time step
def UpdatePotential(N, rho, acc):
	# Initialize the charge grid
	potentialDistribution = GenerateCharge(N, rho)
	
	# Make a deep copy of the charged grid to retain for any 
	# further operations
	charge = np.copy(potentialDistribution)
	
	# Now update the overall plot indefinitely
	count = 0
	increment = 0
	errSum = 10
	while(errSum > acc):
		
		# Make a copy of the initial charge distribution
		potential_init = np.copy(potentialDistribution)	
		
		# Update now the charge distribution through the Gauss-Seidel
		# algorithm manually implemented
		potentialDistribution = ManualGaussSeidel(potentialDistribution, charge)
		
		# Calculate the error sum from the difference of each element before
		# and after the update. The error analysis is implemented to study the
		# convergence of the series
		errSum = np.sum(np.abs(potential_init - potentialDistribution))
		
		if(increment % 20 == 0):
			print("The error sum is: " + str(errSum))
			print(increment)
			
		# Increase the control variable in the loop
		increment += 1
		
	# Save the electric potential in a .csv file
	# new_potential = potentialDistribution.reshape(potentialDistribution.shape[0], -1)
	new_potential = potentialDistribution.flatten()
	df = pd.DataFrame({"Magnetic potential" : new_potential})
	df.to_csv("MagneticPotential.csv")
		
	# In the end, plot the potential through a midplane cut
	plt.imshow(potentialDistribution[:,int(N/2),:])
	plt.show()
	plt.colorbar()

def main():
	# UpdatePotential(50, 1)
	# AnimatePotential(50,1)
	
	# Read the following values from the terminal: the number of
	# cells per each row N, the value of the mid-charge rho and the 
	# accuracy in working out the electric potential acc
	N = int(sys.argv[1])
	j = int(sys.argv[2])
	acc = float(sys.argv[3])
	
	ok = int(input("Introduce the following command: " + "\n" + "(0) Animate the magnetic vector potential" + "\n" 
		+ "(1) Work out the electric potential after reaching the threshold of: " + str(acc) + "\n"
		+ "Place here your command: "))
		
	if(ok == 0):
		AnimatePotential(N,j)
		
	elif(ok == 1):
		UpdatePotential(N, j, acc)
		
	else:
		raise Exception("Introduce an integer between 0 and 1!!")
main()

