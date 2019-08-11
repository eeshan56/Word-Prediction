# Word-Prediction


Installation on Windows:


	1) Install Python 3 (Latest version):
		i) Download from https://www.python.org/downloads/
		ii) Install using downloaded exe file. 
	2) Install pip (Latest version):
		i) Copy code from https://bootstrap.pypa.io/get-pip.py
		ii) Execute get-pip.py (Enter command: python get-pip.py)
	3) Install Anaconda (Latest version):
		i) Download from https://www.anaconda.com/distribution/
		ii) Enter necessary options during installation
	4) Run command "conda create -n tensorflow_env tensorflow" in Anaconda Prompt
	(This creates a virtual environment and installs tensorflow in it)
	
	5) To activate above environment: conda activate tensorflow_env
	To execute Prediction program, always enter above command to activate tensorflow_env

	6) Install gensim, nltk packages:
		Commands:
			i) conda install gensim
				If the program shows C Extension loading error:
				Install C Compiler for faster training. One can install MinGW C compiler. Download setup from https://sourceforge.net/projects/mingw/files/latest/download
				Install all necessary packages in the Setup.
				Add MinGW to Path in System Environment Variables. (Help: https://www.youtube.com/watch?v=sXW2VLrQ3Bs)
			ii) pip install nltk
			Download 'punkt' package of nltk:
			Open a Python Shell (Command:- python) and enter:
				import nltk
				nltk.download('punkt')
			To close Python Shell, enter command: exit() or press Ctrl+Z
			iii) Enter command for training corpus (train.py script):
				python train.py
			iv) Enter command for running the Predict.py script:
				python Predict.py
        
Installation on Linux:


  1) Install tensorflow. Follow the instructions on the website: https://www.tensorflow.org/install/
  2) Install nltk and gensim as mentioned above
