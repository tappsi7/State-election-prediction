Theses three files build the basis of the model the scraping file, scrapes state level polling data for the german state of Saxony-Anhalt and cleans it up. 
The stan file contains the dirichlet model. It models state party support based on national party support times a state offset and includes a house effect as well. 
The estimation py runs the model and saves the draw. 
The combined dataset conatins the current national level polls and the othe csv file containes the state level polls. 
Both of these datasets and the files make it possible to replicate the estimation for german state election in Saxony-Anhalt on the 06th of September 2026.
