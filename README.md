# Overview

This is a repository that contains POI (Point of interest) recommenders and diversifiers.


# Setup


To set up first start with running the Makefile.

	make

Download Yelp Open Dataset (https://www.yelp.com/dataset) and put it files in data folder. Also download https://www.yelp.com/developers/documentation/v3/all_category_list/categories.json and put it in data directory.

After that, it's possible to generate the pre-processed datasets that we use to do evaluations with the following command:

	cd algorithms/	
	python datasetgen.py

Python requirements are specified in the Pipfile.

# Citation

Please if this code is useful in your research consider citing the following papers:

	@article{werneck2021systematic,
	  title={A systematic mapping on POI recommendation: Directions, contributions and limitations of recent studies},
	  author={Werneck, Heitor and Silva, N{\'\i}collas and Carvalho, Matheus and Pereira, Adriano CM and Mour{\~a}o, Fernando and Rocha, Leonardo},
	  journal={Information Systems},
	  pages={101789},
	  year={2021},
	  publisher={Elsevier}
	}
	@article{werneck2021effective,
	  title={Effective and diverse POI recommendations through complementary diversification models},
	  author={Werneck, Heitor and Santos, Rodrigo and Silva, N{\'\i}collas and Pereira, Adriano CM and Mour{\~a}o, Fernando and Rocha, Leonardo},
	  journal={Expert Systems with Applications},
	  volume={175},
	  pages={114775},
	  year={2021},
	  publisher={Elsevier}
	}
