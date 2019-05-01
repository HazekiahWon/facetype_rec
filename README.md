# facetype recognition
## usage
### installation
1. install anaconda  
2. open `conda prompt`, type:  
`git clone https://github.com/HazekiahWon/facetype_rec.git`  
`cd facetype_rec`
3. in `conda prompt`, type:  
`conda install --file requirements.txt`
4. create a folder `ftdata` and put your images there.
### training
1. align data
`python Make_aligndata_git.py`
2. train and cross-validate classifiers
`python Make_classifier_git.py`
### testing
1. By default :
`python realtime_facenet_git.py --rel_path ftdata\polygon`
2. In order to show every image :
`python realtime_facenet_git.py --show_flag 1 --rel_path ftdata\polygon` 
3. if you set `show_flag` to `1`:  
type `q` to quit, and any other key to continue.
4. The test results will be output to `test_results.csv` by default, to set the file path:  
`python realtime_facenet_git.py --rel_path ftdata\polygon --output_file results`  
which sets the output file name to `results.csv`
5. To enable the recommendation mode:  
`python realtime_facenet_git.py --rel_path test --choice 1`  
by default, the relative test data dir is `test` and recommendation mode `1`.  
Equivalently, the above command has the default behavior of `python realtime_facenet_git.py`
### Catching up with the code
go to your installation directory, e.g. `cd facetype_rec`  
`git pull`


