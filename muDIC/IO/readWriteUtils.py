import dill
import csv
import muDIC as dic


def save(dic, name):
    try:
        with open(name + '.p', 'wb') as myfile:
            dill.dump(dic, myfile)
    except:
        raise IOError('Could not save to file')


def load(name):
    try:
        with open(name + '.p', 'rb') as myfile:
            return dill.load(myfile)
    except:
        raise TypeError("Invalid inputs")


def exportCSV(fields,name,frame):
    """ 
    
    Export the field data of a frame to a CSV file
  
    This method exports following field data

    - Corrdinates (x,y)
    - Displacment (ux,uy)
    - True strain (true_strain_xx,true_strain_yy,true_strain_xy)
    - Engineering strain (end_strain_xx,eng_strain_yy,eng_strain_xy)
    - Deformation gradient (F_xx,F_yy,F_xy)

    to the comma seperated value (CSV format). 
  
    Parameters: 
    fields (dic.post.viz.Fields): Fields with the DIC results
    name (string) File name of the CSV file with out the file extension (.csv)
    frame (int) The frame number to export 
  
    """

    if not isinstance(fields, dic.post.viz.Fields):
        raise ValueError("Only instances of Fields are accepted")

    try:
        xs, ys = fields.coords()[0, 0, :, :, frame], fields.coords()[0, 1, :, :, frame]
        # Displacement
        dx = fields.disp()[0, 0, :, :, frame]
        dy = fields.disp()[0, 1, :, :, frame]
        # True strain
        ts_xx = fields.true_strain()[0, 0, 0, :, :, frame]
        ts_yy = fields.true_strain()[0, 1, 1, :, :, frame]
        ts_xy = fields.true_strain()[0, 0, 1, :, :, frame]
        # Engineering strain
        es_xx = fields.eng_strain()[0, 0, 0, :, :, frame]
        es_yy = fields.eng_strain()[0, 1, 1, :, :, frame]
        es_xy = fields.eng_strain()[0, 0, 1, :, :, frame]
        # Deformation gradient
        dg_xx = fields.F()[0, 0, 0, :, :, frame]
        dg_yy = fields.F()[0, 1, 1, :, :, frame]
        dg_xy = fields.F()[0, 0, 1, :, :, frame]

        id = 0
        with open(str(name)+'.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["id","x","y","ux","uy","true_strain_xx","true_strain_yy","true_strain_xy","eng_strain_xx","eng_strain_yy","eng_strain_xy","F_xx","F_yy","F_xy"])
            for i in range(0,len(xs)):
                for j in range(0,len(xs[i])):
                    writer.writerow([id,xs[i][j],ys[i][j],dx[i][j],dy[i][j],ts_xx[i][j],ts_yy[i][j],ts_xy[i][j],es_xx[i][j],es_yy[i][j],es_xy[i][j],dg_xx[i][j],dg_yy[i][j],dg_xy[i][j]])
                    id += 1
    except:
        raise IOError('Could not write CSV files')