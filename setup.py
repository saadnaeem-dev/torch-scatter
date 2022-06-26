from setuptools import setup

setup(
	name="torch_scatter",
	version="1.0",
	author="Saad Naeem",
	packages=['torch_scatter'],
	description= "Minimizes/Maximizes all values from the src tensor into output var at the indices specified in the index tensor Implements scatter_min and scatter_max function.",
	long_description=open("README.md").read(),
)