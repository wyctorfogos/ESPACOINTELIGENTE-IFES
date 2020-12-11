import os

#os.system("python capture-images.py -p1 -g1")   
#os.system("python make-videos.py -p1 -g1")
os.system("python request-2d-skeletons.py -p1 -g1")
os.system("python request-3d-skeletons.py -p1 -g1")
os.system("python export-video-3d-medicoes-erros-no-3d.py -p1 -g1")

