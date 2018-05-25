#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python


import os
os.chdir("/home/binta-asus/Bureau/stageAnnee4/notebook")


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from PIL import Image
from random import shuffle
from random import seed
import shutil



fichier_final =  open("/tmp/csv/mnist_butch_train.csv", "w")
fichier_final2 = open("/tmp/csv/mnist_butch_test.csv", "w")


Names = [['./training-images','train'], ['./test-images','test']]
#Names = [['./training-images','train']]


for name in Names:
    
    FileList = []
#    for dirname in os.listdir(name[0])[1:]: # [1:] Excludes .DS_Store from Mac OS
    for dirname in os.listdir(name[0])[1:]: # [1:] Excludes .DS_Store from Mac OS
        path = os.path.join(name[0],dirname)
        ii=0
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                FileList.append(os.path.join(name[0],dirname,filename))
                ii=ii+1
        print('x shape =',path,ii)
        
    seed()
    shuffle(FileList) # Usefull for further segmenting the validation set


    i_max=0
    for iii in FileList:
        i_max=i_max+1
    print ('i_max= ',i_max) 
    
    
    iiii=0    
    sip=0        
    for iii in FileList:
        if iiii<9*i_max/10:
            shutil.copyfileobj(open(iii, 'r'), fichier_final)
            iiii=iiii+1
        else:
            shutil.copyfileobj(open(iii, 'r'), fichier_final2)
         #   iiii=iiii+1
            sip=sip+1
        
    
    print ('iiii= ',iiii,'/ sip= ',sip)   


fichier_final2.close()
fichier_final.close()

print ('This is the end')
                
                           
#        Im_resized.save(os.path.join( "PNG",filename),"PNG")
