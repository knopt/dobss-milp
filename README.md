BASED ON: Deployed ARMOR Protection: The Application of a Game Theoretic Model for Security at the Los Angeles International Airport  
http://teamcore.usc.edu/papers/2008/AAMASind2008Final.pdf

This program (dobss.py) requires python3.
To create environment with python3:  

``` 
virtualenv -p $(which python3) venv
source venv/bin/activate
pip3 install pulp
```

The following commands might be required to be run beforehand  
```
sudo apt-get install python3-pip
sudo pip3 install virtualenv
```

To run the program with example data run:  
` (venv) python dobss.py <dobss.in`

Example data shows that mixed strategies can be optimal with Stackleberg games.  

To run the program with own data just run:  
` (venv) python dobss.py`