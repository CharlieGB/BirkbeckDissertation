**Implementation of Complementary Temporal Difference Learning model**

Using Stable-Baselines 3

Thanks to Sam Blakeman and Denis Mareschal

requirements.txt contains the modules required by the project

Latest versions of libraries should work, however there may be a need to downgrade numpy<2

MainCode folder contains the Python modules used.

Primary Files

TrainCTDL.py - main code to run models, many test functions used during development
CTDL.py - main code for CTDL model, derived from Stable Baselines DQN class
CTDLPolicy.py - implements policy and predict function using both DQN and SOM contents
SOM.py - implementation of SOM and SOMLayer classes
Maze.py - implementation of Maze in Gymnasium environment
Plotters.py - code for outputting graphics
ExplanationUtils.py - extraction of explanations. Can also be run standalone to extract explanations from previous models.
EventsForModel.py - not used, but can be used to hook into CTDL events