# deep-learning-project

This is the repository for our Deep Learning course in Fall 2020 at University of Utah.

In this project, using neural networks, we will do classification tasks on the Geolife GPS Trajectory Dataset released by Microsoft.

https://msropendata.com/datasets/d19b353b-7483-4db7-a828-b130f6d1f035

The dataset was collected in Microsoft Research Asia Geolife project by 182 users from April 2007 to August 2012. 
A GPS trajectory of this dataset is represented by a sequence of time-stamped points, 
each of which contains the information of latitude, longitude and altitude. Furthermore, 73 users have labeled 
their trajectories with transportation mode as car, bus, walk or train.

Team Members: Hasan Pourmahmoodaghababa, Ali Hassandokht Mashhadi, Behnam Sheraafat


In November 29, we uploaded the mapped data from our trajectory data to R^20, R^50 and R^250 under two different feature 
mappings appeared in our refrences [1,2]. They are stored in folders Mapped-Data, Mappead-Data-50-dim and Mapped-Data-250-dim, respectively.

Landmarks employed for creating these datasets are stored in Landmarks folder in csv format. The first one with 
size 20 is handpiched from near the mean and median of waypoints of the trajectories. The second one with size 50 
is handpicked from a map visualization of all trajectories in such a way that they are chosen to be near to a group 
of trajectores. The last one with size 250 is constructed by 200 gridding points plus previously chosen 50 points from the map. 

The preprocessing codes in python language are uploaded in the Pre-processing-codes folder. 
