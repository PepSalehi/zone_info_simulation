import argparse

parser = argparse.ArgumentParser(description="Simulation of drivers' behavior")
parser.add_argument('-f', '--fleet', 
						help='Fleet sizes to simulate, formatted as comma-separated list (i.e. "-f 250,275,300")')

args = parser.parse_args()
if args.fleet:
    fleet_size = int(args.fleet)
print("fleet_size ", fleet_size )
print(type(fleet_size))