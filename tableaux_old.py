import numpy as np
from pymatching import Matching
import random as rnd

def get_vertex_row(k,Nx,Ny): 
    '''
    Find the row number of the vertex k

    k = Nx*div(k,Nx) + mod(k,Nx)
    Input:
        k : vertex index
        Nx : linear dimensions in x-direction
    '''
    assert k<Nx*Ny
    return k//Nx

def get_vertex_column(k,Nx):
    '''
    Find the column number of the vertex k
    k = Nx*div(k,Nx) + mod(k,Nx)
    Input:
        k : vertex index
        Nx : linear dimensions in x-direction
    '''
    return k%Nx

def star_edges_of_vertex(k,linear_size): 
    '''
    Find the edges where X acts in the star at vertex k (A_k)
    Input:
        k : vertex index
        linear_size=(Nx,Ny) : linear dimensions as a tuple
    Output:
        mm : 6 element array. counterclockwise order starting the top horizontal edge
    '''
    Nx = linear_size[0]
    Ny = linear_size[1]
    assert k<Nx*Ny and k>=0 , "Error: Vertex index is out of bounds"

    row = get_vertex_row(k,Nx,Ny)
    col = get_vertex_column(k,Nx)

    next_row = Nx*((row+1)%Ny)
    prev_col = (col-1)%Nx
    m1 = 2*Nx*Ny + Nx*((row-1)%Ny) + prev_col
    m2 = Nx*row + prev_col
    m3 = Nx*Ny + next_row + prev_col
    m4 = 2*Nx*Ny + next_row + col
    m5 = next_row + (col+1)%Nx
    m6 = Nx*Ny +k

    return np.array([m1,m2,m3,m4,m5,m6])

def ordered_star_edge_matrix(linear_size): 
    '''
    Return the X edges of star in ordered fashion
    '''
    Nx,Ny = linear_size
    op = np.zeros((Nx*Ny,6),dtype=int)
    for i in range(Nx*Ny):
        op[i,:] = star_edges_of_vertex(i,(Nx,Ny))
    
    return op

def star_check_matrix(linear_size): 
    '''
    X operators of star (A_s) matrix on the Kagome lattice
    Input:
        Nx : Linear x-dimension
        Ny: Linear y-dimension
    Output:
        HH: check matrix for star stabilizer. HH[i,j] = 1 iff hexagon A centered at i has X operator on edges j
    '''
    Nx,Ny = linear_size

    HH = np.zeros((Nx*Ny,3*Nx*Ny),dtype=np.uint8)

    for k in range(Nx*Ny):
        mm = star_edges_of_vertex(k, (Nx,Ny))
        assert mm.shape==(6,)
        HH[k,mm]=1
    return HH

def get_edge_color_array(linear_size): 
    '''
    Find the color of edges. 0: red , 1: green, 2: blue
    Input:
        linear_size
    Output:
        arr: arr[j] = {0,1,2} where j is edge index
        check readme for labeling convention
    '''
    Nx,Ny = linear_size
    arr = np.zeros(3*Nx*Ny, dtype=np.int64)

    cc_now = 1 #start with green for type-1 edges
    for i in range(Nx*Ny):
        arr[i] = cc_now
        if i%Nx==(Nx-1):
            cc_now = (cc_now+2)%3
        else:
            cc_now = (cc_now+1)%3
    
    cc_now = 0 #start with red for type-2 edges
    for i in range(Nx*Ny,2*Nx*Ny):
        arr[i] = cc_now
        if i%Nx==(Nx-1):
            cc_now = (cc_now+2)%3
        else:
            cc_now = (cc_now+1)%3
    
    cc_now = 2 #start with blue for type-3 edges
    for i in range(2*Nx*Ny,3*Nx*Ny):
        arr[i] = cc_now
        if i%Nx==(Nx-1):
            cc_now = (cc_now+2)%3
        else:
            cc_now = (cc_now+1)%3

    for i in range(3):
        assert np.count_nonzero(arr==i)==(Nx*Ny)
    return arr 

def bL_edges_of_vertex(k,linear_size): 
    '''
    Find the edges corresponding to the B operator:  >>>(k)---
    the two edges point to left >- shape
    |> triangles
    Input:
        k : vertex index
        linear_size=(Nx,Ny) : linear dimensions as a tuple
    Output:
        mm : 3 element array. counterclockwise order starting the top edge
    '''
    Nx = linear_size[0]
    Ny = linear_size[1]
    assert k<Nx*Ny and k>=0 , "Vertex index is out of bounds"

    row = get_vertex_row(k,Nx,Ny)
    col = get_vertex_column(k,Nx)
    m1 = Nx*Ny + Nx*row + (col-1)%Nx
    m2 = Nx*((row+1)%Ny) + col
    m3 = 2*Nx*Ny + k
    return np.array([m1,m2,m3])

def bR_edges_of_vertex(k,linear_size): 
    '''
    Find the edges corresponding to the B operator:  ----(k)<<<<
    the two edges point to right -< shape
    <| triangles
    Input:
        k : vertex index
        linear_size=(Nx,Ny) : linear dimensions as a tuple
    Output:
        mm : 3 element array. counterclockwise order starting the top edge
    '''
    Nx = linear_size[0]
    Ny = linear_size[1]
    assert k<Nx*Ny and k>=0 , "Vertex index is out of bounds"

    row = get_vertex_row(k,Nx,Ny)
    col = get_vertex_column(k,Nx)

    m1 = k
    m2 = 2*Nx*Ny + Nx*row + (col-1)%Nx
    m3 = Nx*Ny + Nx*((row+1)%Ny) + col
    return np.array([m1,m2,m3])

def ordered_bl_edge_matrix(linear_size): 
    '''
    Return the edge numbers of BL ordered by vertex k
    '''
    Nx,Ny = linear_size
    op = np.zeros((Nx*Ny,3),dtype=int)
    for i in range(Nx*Ny):
        op[i,:] = bL_edges_of_vertex(i,(Nx,Ny))    
    return op

def ordered_br_edge_matrix(linear_size): 
    '''
    Return the edge numbers of BR ordered by vertex k
    '''
    Nx,Ny = linear_size
    op = np.zeros((Nx*Ny,3),dtype=int)
    for i in range(Nx*Ny):
        op[i,:] = bR_edges_of_vertex(i,(Nx,Ny))
    return op

def triangle_check_matrix(linear_size, which): 
    '''
    Triangular (Br: ==<<, <| triangle)  or (BL:  >>====, |> triangle) check matrix on the Kagome lattice                            

    Input:
        Nx : Linear x-dimension
        Ny: Linear y-dimension
        which: 'left' or 'right'
    Output:
        HH: parity check matrix for triangle stabilizer. HH[i,j] = 1 iff B centered at i has edges j
    '''
    Nx,Ny = linear_size
    assert Nx>0
    assert Ny>0
    assert which=='right' or which=='left',"Incorrect specification for Br or BL"

    HH = np.zeros((Nx*Ny,3*Nx*Ny),dtype=np.uint8)
    for k in range(Nx*Ny):

        if which=='right':
            mm = bR_edges_of_vertex(k, (Nx,Ny))
        elif which=='left':
            mm = bL_edges_of_vertex(k, (Nx,Ny))
        assert mm.shape==(3,)
        HH[k,mm]=1

    return HH

def z_logical_check_matrix(linear_size):  
    '''
    see readme for the layout
    Input:
        linear_size
    Output:
        HH = (12,3*Nx*Ny) shaped array for z-logical parity check matrix
        [Rv,Gv,Bv,Rh,Gh,Bh, (Rv,Gv,Bv,Rh,Gh,Bh)'  ]
        The last six primed operators are to be used for updating logical X operators, but they are equivalent to the unprimed counterpart
    '''
    Nx,Ny=linear_size
    HH = np.zeros((12,3*Nx*Ny),dtype=np.uint8)
    # Rv Gv Bv
    for i in range(3):
        for j in range(0,Ny,3):
            HH[i,Nx*Ny+ Nx*j+i] = 1
            HH[i,2*Nx*Ny + Nx*(j+1)+i] = 1
    # primed: Rv Gv Bv
    for i in range(6,9):
        for j in range(0,Ny,3):
            HH[i,2*Nx*Ny + Nx*j + (i+1)%3] = 1
            HH[i,Nx*Ny + Nx*(j+2) + (i+1)%3] = 1

    #Rh, Gh, Bh
    for i in range(3,6):
        for j in range(0,Nx,3):
            HH[i,Nx*Ny + Nx*(i%3) + j] = 1
            HH[i, Nx*(i%3) + j+2] = 1
    #primed: Rh, Gh, Bh
    for i in range(9,12):
        for j in range(1,Nx,3):
            HH[i,Nx*((i+1)%3) + j] = 1
            HH[i, Nx*Ny + Nx*((i+1)%3) + j+1] = 1
    assert np.array_equal(np.count_nonzero(HH, axis=1),
                        np.array([2*Ny/3,2*Ny/3,2*Ny/3,2*Nx/3,2*Nx/3,2*Nx/3]*2))\
        , print(np.count_nonzero(HH, axis=1),'!=',np.array([2*Ny/3,2*Ny/3,2*Ny/3,2*Nx/3,2*Nx/3,2*Nx/3]*2))
    return HH

def x_bare_logical_check_matrix(linear_size): 
    '''
    X operators in the logical X. This does not include the additional control-z operations
    [Rv,Gv,Bv, Rh,Gh,Bh]
    Returns:
        (6,Nq) shaped array: parity check matrix for bare x logical (x logical of toric code)
    '''
    Nx,Ny=linear_size
    HH = np.zeros((6,3*Nx*Ny),dtype=np.uint8)

    # X Vertical:
    ind = np.array([2,0,1])
    for i in range(3):
        col = Nx*np.arange(0,Ny,3) + ind[i]
        HH[i,col] = 1       
        col = Nx*Ny +  Nx*np.arange(1,Ny,3) + ind[i]
        HH[i,col] = 1
        col = 2*Nx*Ny + Nx*np.arange(2,Ny,3) + ind[i]
        HH[i,col] = 1
        if Nx == 3:
            col = Nx*np.arange(2,Ny,3) + (1+ ind[i])%3
        else: 
            col = Nx*np.arange(2,Ny,3) + (1+ ind[i])
        HH[i,col] = 1
    # X horizontal:
    for i in range(3,6):
        for j in range(0,Nx,3):
            HH[i, Nx*((i+2)%3)+j]=1
            HH[i,Nx*Ny + Nx*((i+2)%3)+j+1]=1
            HH[i,2*Nx*Ny + Nx*((i+2)%3)+j+2]=1
            HH[i,2*Nx*Ny + Nx*((((i+2)%3)-1)%Ny)+j]=1
    assert np.array_equal(np.count_nonzero(HH, axis=1),np.array([4*Ny/3,4*Ny/3,4*Ny/3,4*Nx/3,4*Nx/3,4*Nx/3]))\
        , print(np.count_nonzero(HH, axis=1))
    return HH

def ground_state_stabilizer(linear_size,edge_color_arr):
    '''
    Construct the ground state stabilkzier matrix

    first 3*Nx*Ny colums = keep track of Z (absent in ground state will appear later as decorations)
    remaining coloumns =1 whenever

    jth row = j^th stabilizer
    SS[j,k] = 1 if Z_k is present for 0<=k<=(3*Nx*Ny-1) in j^th stab = 0 in ground state
    SS[j,k] = 1 if A_{k-(3*Nx*Ny)} is present in j^th stab = 1 
    in ground state k^st row is A_k and so on

    last three rows represent product condition prod_c A_c=1 for each color
    this is accounted by seeting all A of same color = 1 in last three rows
    order from top to bottom [b,r,g]=[2,0,1]
    Parameters
    ----------
    linear_size : 
        (Nx,Ny)
    edge_color_arr : 
        edge_color_arr[k] = color of edge k = {0,1,2}

    Returns
    -------
    SS
        array
    '''
    Nx,Ny=linear_size
    SS = np.zeros((Nx*Ny,4*Nx*Ny),dtype=np.uint8)
    
    cc = edge_color_arr[Nx*Ny:2*Nx*Ny]
    for i in range(SS.shape[0]):
        SS[i,i+3*Nx*Ny] = 1
    
    cc_row = [Nx*Ny-2 , Nx*Ny-1, Nx*Ny-3]
    for i in range(3):
        cols = np.arange(Nx*Ny)[(cc==i)]
        SS[cc_row[i],3*Nx*Ny+cols]=1
    return SS

def ground_state_destabilizer(linear_size,HH_star): 
    '''
    Compute the destabilizer matrix for the D4 ground state
    {D_k,A_k} = 0, [D_k,A_j]=0 if j not k
    constructed by pairing D_k with pre-defined vertex of same color=c using string of Z on dual lattice of c-honeycomb
    Parameters
    ----------
    linear_size : 
        (Nx,Ny)
    HH_star : 
        parity check matrix for  A stabilizers (ignoring CZ gates)

    Returns
    -------
    DD
        (Nx*Ny-3,Nx*Ny*4) shaped array
        [:3*Nx*Ny]=1 if Z is present at the location
    '''
    Nx,Ny=linear_size
    DD = np.zeros((Nx*Ny-3,4*Nx*Ny))
    hole_list = [Nx*Ny-2,Nx*Ny-1,Nx*Ny-3] #['r','g','b'], location of the reference A_p*
    count = 0 #keeps track of color of the A oeprator
    matching_graph = Matching.from_check_matrix(HH_star)
    for row in range(DD.shape[0]):
        syndrome = np.zeros(Nx*Ny)
        syndrome[row] = 1
        syndrome[hole_list[count]]=1
        prediction = matching_graph.decode(syndrome)
        DD[row,0:3*Nx*Ny] = prediction
        if row%Nx==Nx-1:
            count = (count-1)%3
        else:
            count = (count+1)%3
    return DD

def connection_dict(linear_size):
    '''
    Construction relevant matrices that contain details about the connectivity on lattice
    Input:
        linear size
    Returns:
        out : dict
        HH_star: X edges of A star in the form of a matrix
        HH_log_z: Z edges of logical Z operators in the form of a matrix, 2 copies
        HH_log_x: X edges of logical X operators in the form of a matrix
        HH_br, HH_bl: Z edges of triangles in the form of a matrix
        bR_stored_edges, bL_stored_edges: edge numbers of the triangles
        star_stored_edges: edge numbers of the X edges of A stars
        edge_color_arr: color of edges, 0: red , 1: green, 2: blue
        hex_corners: star neighbors of each site k, with whom star at k commutes up to 2 triangles
        star_syndromes: star_syndromes[k,:] = stars that have overlapping X on edge k
        mgraph: matching graph for 0: red triangle, 1: green triangle, 2: blue triangle, 3: all stars
        xLog_CZ: dictionary[key] = (x_log_p, list of edges that get dressed with Z in x_log_p), key is the edge number on which X acts
        bottom_dict: star at kv commutes with x_log_ind up to z_log_ind, returns dict={(x_log_ind,kv): z_log_ind}
        bR_vertex_color_arr, bL_vertex_color_arr, star_color_arr: color of triangles or stars at each site k, 0: red , 1: green, 2: blue
        endpoint_vertex: list of dictionaries, index is the edge number on which X acts, 'pv' 'qv' are affected stars, 'pe' 'qe' are Z dressings on 'pv' 'qv'
        A_xlog_comm_bl, A_xlog_comm_br: (x_log,A_kv):Br_kv, the key is the X logical and star A, returns the triangle up to which the local and star commute
    '''
    Nx,Ny = linear_size
    HH_star = star_check_matrix(linear_size)
    HH_log_z = z_logical_check_matrix(linear_size)
    HH_log_x = x_bare_logical_check_matrix(linear_size)
    HH_br = triangle_check_matrix(linear_size,'right')
    HH_bl = triangle_check_matrix(linear_size,'left')
    out = {}
    out['HH_star'] = HH_star.copy()
    out['HH_log_z'] = HH_log_z.copy()
    out['HH_log_x'] = HH_log_x.copy()
    out['HH_br'] = HH_br.copy()
    out['HH_bl'] = HH_bl.copy()
    out['bR_stored_edges'] = ordered_br_edge_matrix(linear_size)
    out['bL_stored_edges'] = ordered_bl_edge_matrix(linear_size)
    out['star_stored_edges'] = ordered_star_edge_matrix(linear_size)
    out['edge_color_arr'] = get_edge_color_array(linear_size)

    vv = np.zeros((Nx*Ny,6),dtype=int)
    for i in range(Nx*Ny):
        vv[i,:] = find_hexagon_corners(i,linear_size)
    out['hex_corners']=vv
    
    star_v = np.zeros((3*Nx*Ny,2),dtype=int)
    for i in range(3*Nx*Ny):
        star_v[i,:] = HH_star[:,i].nonzero()[0]
    out['star_syndromes'] = star_v #star_syndromes[k,:] = stars that share edge k, i.e. these two stars have overlapping X on edge k

    mgraph = generate_mgraph(HH_star,HH_br,HH_bl,out['edge_color_arr'])
    out['mgraph'] = mgraph

    out['xLog_CZ'] = z_decoration_on_LX(linear_size,out['edge_color_arr'])
    out['bottom_dict'] = construct_bottom_dictionary(linear_size)

    out['bL_vertex_color_arr'] = np.hstack((out['edge_color_arr'][Nx:Nx*Ny],out['edge_color_arr'][0:Nx]))
    out['bR_vertex_color_arr'] = out['edge_color_arr'][0:Nx*Ny]
    out['star_color_arr'] = out['edge_color_arr'][Nx*Ny:2*Nx*Ny]

    endpt_list=[]
    tmp_out = {'bR_stored_edges':out['bR_stored_edges'],'bL_stored_edges': out['bL_stored_edges']}
    for i in range(3*Nx*Ny):
        endpt_list.append(find_endpoint_vertex(i,linear_size,tmp_out))
    out['endpoint_vertex']=endpt_list

    A_xlog_comm_bl,A_xlog_comm_br=A_xlog_comm_dict(linear_size)
    out['A_xlog_comm_bl']=A_xlog_comm_bl
    out['A_xlog_comm_br']=A_xlog_comm_br
    return out

def find_hexagon_corners(k,linear_size):
    '''
    Find the indices of the 6 star neighbors of the star at vertex k, the vertex at k commutes with each neighbor up to 2 triangles
    ordering: 0 starts at left endpoint of bottom edge (south-west corner), counterclockwise

          4 ===== 3
        //         \\
       //           \\
      5       k       2
        \\          //
         \\        //
           0 ===== 1

    Input:
        k : vertex index 0<=k<=(Nx*Ny)
        linear_size: (Nx,Ny)
    Returns: (6,) shaped array
    '''
    Nx,Ny = linear_size
    assert k<Nx*Ny and k>=0 , "Error: Vertex index is out of bounds"
    row = get_vertex_row(k,Nx,Ny)
    col = get_vertex_column(k,Nx)

    v0 = Nx*((row+1)%Ny) + col
    v1 = Nx*((row+1)%Ny) + (col+1)%Nx
    v2 = Nx*row + (col+1)%Nx
    v3 = Nx*((row-1)%Ny)+col
    v4 = Nx*((row-1)%Ny) + (col-1)%Nx
    v5 = row*Nx + (col-1)%Nx
    
    return np.array([v0,v1,v2,v3,v4,v5])

def generate_mgraph(HH_star,HH_br,HH_bl,edge_color_arr): 
    '''
    Generate matching graph objects for all check operators
    Input:
        HH_star: star check matrix
        HH_br : Br check matrix
        HH_bl, Bl check matrix
    Output:
        mgraph_dict = {0: mgrpah_red_B, 1: mgraph_green_B, 2: mgraph_blue_B, 3: mgraph_star}
    '''
    mgraph_dict = {}
    mgraph_dict[3] = Matching.from_check_matrix(HH_star)

    for cc in range(3):
        HH_cc = (HH_bl + HH_br)*(edge_color_arr==cc)
        mgraph_dict[cc] = Matching.from_check_matrix(HH_cc) 

    return mgraph_dict

def z_decoration_on_LX(linear_size,edge_color_arr):
    '''
    Decoration of logical X by Z when X acts on qubits that participate in the control-z gates
    
    Return dictionary with dd[ke] = (x_log_p, list of edges that get decorated in x_log_p)
    if X_{dd[ke]} is applied it decorates x_log[dd[ke][0]] at edges labeled by x_log[dd[ke][1]] by Z operator

    x_log_p is 0 to 5
    '''
    Nx,Ny = linear_size
    top=[1,2,0]
    bottom=[2,0,1]
    left=[1,2,0]
    right=[2,0,1]
    dd={}
    for k in range(3*Nx*Ny):
        for p in range(3):
            col = get_vertex_column(k%(Nx*Ny),Nx)
            row = get_vertex_row(k%(Nx*Ny),Nx,Ny)
            if edge_color_arr[k]==top[p] and col==(p+2)%3 and k//(Nx*Ny)!=0:                
                l1 = Nx*Ny + Nx*np.arange(0,row+1,3)+col
                l2 = 2*Nx*Ny + Nx*np.arange(1,row,3)+col
                if k in dd:
                    dd[k] = dd[k]+[(p,np.hstack((l1,l2)))]
                else:
                    dd[k]=[(p,np.hstack((l1,l2)))]
            if edge_color_arr[k]==bottom[p] and col==(p+2)%3 and k//(Nx*Ny)!=0:
                l1 = Nx*Ny + Nx*np.arange(Ny-1,row,-3)+col
                l2 = 2*Nx*Ny + Nx*np.arange(Ny-3,row-1,-3)+col
                if k in dd:
                    dd[k] = dd[k] + [(p,np.hstack((l1,l2)))]
                else:
                    dd[k]=[(p,np.hstack((l1,l2)))]
            
            ##horizontal: towards left
            if edge_color_arr[k]==left[p] and row==(p+2)%3 and k//(Nx*Ny)!=2:
                l1 = Nx*row + np.arange(2,col+1,3)
                l2 = Nx*Ny + Nx*row + np.arange(0,col,3)
                if k in dd:
                    dd[k] = dd[k] + [(p+3,np.hstack((l1,l2)))]
                else:
                    dd[k]=[(p+3,np.hstack((l1,l2)))]
            if edge_color_arr[k]==right[p] and row==(p+2)%3 and k//(Nx*Ny)!=2:
                l1 = Nx*row + np.arange(Nx-2,col,-3)
                l2 = Nx*Ny + Nx*row + np.arange(Nx-1,col-1,-3)
                if k in dd:
                    dd[k] = dd[k] + [(p+3,np.hstack((l1,l2)))]
                else:
                    dd[k]=[(p+3,np.hstack((l1,l2)))]
    assert len(dd)==4*(Ny+Nx)-4 ,print(len(dd),'!=',4*(Ny+Nx)-4)
    return dd 

def construct_bottom_dictionary(linear_size): 
    '''
    [A,xlog] = 
    Construct the dictionary listing the z logical operators appearing in the [A,xLog] relation

    Returns dict={(x_log_ind,kv): z_log_ind}
    This function is required because when A at the bottom of lattice is commuted through vertical X_log, it depends on logical Z (in addition to B oeprators)    
    similarly for horizontal X, A on the righmost column will involve logical Z
    '''
    Nx,Ny = linear_size
    dd = {(0,Nx*(Ny-1)+2):2,    (0,Nx*(Ny-1)+3%Nx):7,  (1,Nx*(Ny-1)):0,    (1,Nx*(Ny-1)+1):8,\
        (2,Nx*(Ny-1)+1):1,  (2,Nx*(Ny-1)+2):6,  (3,Nx):5,   (3,2*Nx):10,    (4,Nx*(Ny-1)):3,(4,0):11,\
        (5,0):4 ,(5,Nx):9}
    assert len(dd)==12
    return dd

def find_endpoint_vertex(k,linear_size,cn_dict):
    '''
    Find the index of two stars (pv,qv) at the endpoint of the edge k and 
    the edge indices (pe,qe) of Z decorations that should be added if X_k is applied

    Input:
        k : edge index 0<= k <= (3*NxNy-1)
        linear_size = (Nx,Ny)
        cn_dict: bR and bL stored edges are used
    Returns:
        dictionary: check figure for notations
    '''
    Nx = linear_size[0]
    Ny = linear_size[1]
    
    if k//(Nx*Ny)==0:
        p = (k-Nx)%(Nx*Ny)
        q = k

        zp = cn_dict['bR_stored_edges'][p,[1,2]]
        zq = cn_dict['bL_stored_edges'][q,[0,2]]
    elif k//(Nx*Ny)==1:
        p = ((k-Nx*Ny)//Nx)*Nx + ((k%Nx)+1)%Nx
        q = (((k-Nx*Ny)//Nx-1)%Ny)*Nx + k%Nx

        zp = cn_dict['bR_stored_edges'][p,[0,1]]
        zq = cn_dict['bL_stored_edges'][q,[1,2]]
    elif k//(Nx*Ny)==2:
        p = k-2*Nx*Ny
        q = Nx*(p//Nx) + (p+1)%Nx

        zp = cn_dict['bR_stored_edges'][p,[0,2]]
        zq = cn_dict['bL_stored_edges'][q,[0,1]]
    else:
        print('Error: edge index k is out of bound')
    return {'pv':p,'qv':q,'pe':zp,'qe':zq}

def A_xlog_comm_dict(linear_size):
    '''
    return dictionary containing the location of vertex operator that appears in the 
    commutation between  A_{kv} with 6 x-logical operators
    (x_log,A_kv):Br_kv

    out: bl_out,br_out
    '''
    Nx,Ny = linear_size
    br_out = {}
    bl_out = {}
    for row in np.arange(1,Ny,3):
        br_out[(0,row*Nx + 3%Nx)]=row*Nx + 3%Nx
        br_out[(1,row*Nx + 1)]=row*Nx + 1
        br_out[(2,row*Nx + 2)]=row*Nx + 2
    
    for row in np.arange(2,Ny,3):
        br_out[(0,row*Nx + 2)]=row*Nx + 2
        br_out[(1,row*Nx + 0)]=row*Nx + 0
        br_out[(2,row*Nx + 1)]=row*Nx + 1
    del row

    for col in np.arange(0,Nx,3):
        br_out[(3,1*Nx + col)]=1*Nx + col
        br_out[(4,(Ny-1)*Nx + col)]=(Ny-1)*Nx + col
        br_out[(5,0*Nx + col)]=0*Nx + col
    
    for col in np.arange(2,Nx,3):
        br_out[(3,2*Nx + col)]=2*Nx + col
        br_out[(4,0*Nx + col)]=0*Nx + col
        br_out[(5,1*Nx + col)]=1*Nx + col
    del col

    for row in np.arange(0,Ny,3):
        bl_out[(0,row*Nx + 2)]=row*Nx + 2
        bl_out[(1,row*Nx + 0)]=row*Nx + 0
        bl_out[(2,row*Nx + 1)]=row*Nx + 1
    for row in np.arange(2,Ny,3):
        bl_out[(0,row*Nx + 3%Nx)]=row*Nx + 3%Nx
        bl_out[(1,row*Nx + 1)]=row*Nx + 1
        bl_out[(2,row*Nx + 2)]=row*Nx + 2
    del row
    
    for col in np.arange(0,Nx,3):
        bl_out[(3,2*Nx + col)]=2*Nx + col
        bl_out[(4,0*Nx + col)]=0*Nx + col
        bl_out[(5,1*Nx + col)]=1*Nx + col
    for col in np.arange(1,Nx,3):
        bl_out[(3,1*Nx + col)]=1*Nx + col
        bl_out[(4,(Ny-1)*Nx + col)]=(Ny-1)*Nx + col
        bl_out[(5,0*Nx + col)]=0*Nx + col
    del col
    assert len(bl_out)==(3*(2*Nx/3) + 3*(2*Ny/3) )
    assert len(br_out)==(3*(2*Nx/3) + 3*(2*Ny/3) )

    return bl_out,br_out

def apply_X(k,bL_state,bR_state,SS,DD,RR,LZ_state,LX_vec,LX_sign,linear_size,cn_dict):  
    '''
    Apply X on edge k and update the stabilizer matrices and the logical state of the system

    Parameters
    ----------
    k :
        edge index where X_k is applied
    bL_state : 
        current state of bL
    bR_state :
        current state of bR
    SS : 
        A stabilizer matrix
    DD : 
        Destabilizer matrix for A
    RR : 
        sign of SS rows
    LZ_state : 
        z logical state
    LX_vec : 
        x logical vectors
    LX_sign : 
        signs of x logical        
    linear_size : 
        (Nx,Ny)
    cn_dict : 
        precomuted dictionary 
    '''
    z_logical_check_matrix = cn_dict['HH_log_z'].copy()
    assert k<3*linear_size[0]*linear_size[1]
    assert(SS.shape==(linear_size[0]*linear_size[1],4*linear_size[0]*linear_size[1]))
    assert(DD.shape==(linear_size[0]*linear_size[1]-3,4*linear_size[0]*linear_size[1]))
    assert RR.shape==(SS.shape[0],)
    dc = cn_dict['endpoint_vertex'][k]
    #update B
    bL_state[dc['pv']] = (bL_state[dc['pv']]+1)%2 # p and q orientation is fixed wrt Br and Bl
    bR_state[dc['qv']] = (bR_state[dc['qv']]+1)%2

    #update A
    for j in range(SS.shape[0]): # 0 to Nx*Ny-1, loop through all stabilizers
        update_X_on_SS_row(linear_size,SS,RR,j,dc['pv'],dc['pe'],cn_dict)
        update_X_on_SS_row(linear_size,SS,RR,j,dc['qv'],dc['qe'],cn_dict)
        #flip the sign if Z_k is present
        RR[j] = (RR[j] + SS[j,k])%2
    
    #update destabilizers similar to A
    foo = np.zeros(DD.shape[0]) #proxy for RR unused
    for j in range(DD.shape[0]): # 0 to Nx*Ny-4, loop through all destabilizers
        update_X_on_SS_row(linear_size,DD,foo,j,dc['pv'],dc['pe'],cn_dict)
        update_X_on_SS_row(linear_size,DD,foo,j,dc['qv'],dc['qe'],cn_dict)
    
    #update logical Z
    for lz_flip in z_logical_check_matrix[:,k].nonzero()[0]:
        LZ_state[lz_flip] =(LZ_state[lz_flip]+1)%2

    #update logical x: A part of the operator
    for j in range(LX_vec.shape[0]): #loop through 6 logical Xs
        if LX_sign[j]<=1: #only update active logical Xs, otherwise we are simulating z_logical state
            update_X_on_SS_row(linear_size,LX_vec,LX_sign,j,dc['pv'],dc['pe'],cn_dict)
            update_X_on_SS_row(linear_size,LX_vec,LX_sign,j,dc['qv'],dc['qe'],cn_dict)
            #sign due to Zk already existing in the logical stabilizer
            LX_sign[j] = (LX_sign[j] + LX_vec[j,k])%2
    #update logical x: original XLog part (string bare X as in toric code)
    #this is true when the stabilizers are products in the following order logical*Zs*stars
    try:
        z_decoration_ind = cn_dict['xLog_CZ'][k] #list of tuples (x_log index, list of edges that get dressed with Z in x_log)
        for ll in z_decoration_ind: #loop through all affected logical Xs
            if LX_sign[ll[0]]<=1: #proceed if logical X is active
                LX_vec[ll[0],ll[1]] = (LX_vec[ll[0],ll[1]]+1)%2 #dress Z
    except KeyError:
        pass    

def update_X_on_SS_row(linear_size,SS,RR,row,pv,z_edges,cn_dict):
    '''
    Update the j=row stabilizer at star pv given locations of Z dressings due to Pauli X operation
    This does not account for sign change due to acting by X on Z that is present from before
    Input:
        linear_size
        SS : current star stabilizer matrix
        RR : star stabilizer sign vector [000100001...]
        row : row index, corresponds to the index of stabilizers
        pv : vertex index        
        z_edges : decoration edges
        spoke_ind : 0<=np.array([a,b,c])<=5 -- included in cn_dict
    Returns:
        None
        modify SS,RR
    '''
    Nq = 3*linear_size[0]*linear_size[1]
    foo = SS[row].copy() #new row
    if foo[Nq+pv]==1: #if star pv is in the stabilizer, add Z dressings
        for z in z_edges: #z_edge: the position of Z dressings
            foo[z] = (foo[z]+1)%2 #dress the edge
            # sign change upon commuting Z through X of stars on left (if they exist in row) of pv
            # this sign change comes from the convention that Z and A operators multiply in the same order as the columns for the stabilizers
            stars_to_left = cn_dict['star_syndromes'][z,:]
            stars_to_left = stars_to_left[stars_to_left<pv]
            RR[row] = (RR[row] + np.count_nonzero(foo[Nq+stars_to_left]))%2
    SS[row]=foo #update row
    
def apply_Z(k,linear_size,SS,RR,LX_vec,LX_sign,cn_dict): 
    '''
    modify the stabilizers after application of Z on edge k
    sign of A stabilizer is updated in terms of R
    Modify logical X operator
    
    Parameters
    ----------
    k : 
        edge index
    linear_size : 
        (Nx,Ny)
    SS : 
        current star stabilizer matrix
    RR : 
        sign vector
    LX_vec : 
        logical X non-trivial part (z,A operators)
    LX_sign : 
        sign of logical X
    x_logical_check_matrix : 
        logical-x parity check matrix (similar to toric code)
    cn_dict : 
        precomputed 
    '''
    x_logical_check_matrix = cn_dict['HH_log_x'].copy()
    # if Lx sign is >1 then we are currently in z logical state an no need to update this operator
    assert RR.shape==(SS.shape[0],)
    Nx,Ny=linear_size

    # update sign of stars
    for j in range(SS.shape[0]): #loop through all stabilizers
        for itr in cn_dict['star_syndromes'][k,:]: #find the two affected stars
            if SS[j,itr+3*Nx*Ny] == 1: 
                RR[j] = (RR[j]+1)%2
    # action on original xLog
    for lx_flip in x_logical_check_matrix[:,k].nonzero()[0]: #find all X logicals that have X on k
        if LX_sign[lx_flip]<=1:
            LX_sign[lx_flip] = (LX_sign[lx_flip]+1)%2
    # action on A part of xLog
    for j in range(LX_vec.shape[0]): #loop through 6 logical Xs
        if LX_sign[j]<=1: #update active logical Xs
            for itr in cn_dict['star_syndromes'][k,:]:
                if LX_vec[j,itr+3*Nx*Ny] == 1: 
                    LX_sign[j] = (LX_sign[j]+1)%2

def measure_A(kv,linear_size,SS,DD,RR,bL_state,bR_state,LZ_state,LX_vec,LX_sign,cn_dict):     
    '''
    Update the stabilizer tableau post measurement and return the measurement outcome of A_k star operator
    Input:
        kv : 0<=k<=Nx*Ny ; vertex index A_k is measured
        linear_size: (Nx,Ny)
        SS : current state of A tableau
        DD: Current state of destabilizers 
        RR : current signs of A [010101....]
        bL_state: current state of Bl [0101010...]
        bR_state: current state of Br [01010...]
        LZ_state: logical z state
        LX_vec: non-trivial operator part of logical X
        LX_sign: sign of logical X
        cn_dict
    Returns:
        m_outcome : 0 or 1; modifies SS,RR,DD
        terminate = True: return 5 if LX is no longer a stabilizer, no modification
    '''    
    Nx,Ny = linear_size
    assert kv<Nx*Ny and kv>=0 , "Error: Vertex index is out of bounds"
    assert SS.shape==(Nx*Ny,4*Nx*Ny)
    assert DD.shape==(Nx*Ny-3,4*Nx*Ny)
    
    A_row = np.zeros(4*Nx*Ny,dtype=np.uint8)
    A_row[(3*Nx*Ny)+kv] = 1 #create the stabilizer row corresponding to only the star to be measured
    assert np.array_equal(np.nonzero(A_row)[0],[kv+(3*Nx*Ny)])
    is_generator = np.nonzero(np.all(SS==A_row,axis=1))[0] #index of SS row equals to A_row
    if np.size(is_generator)==1: # A_k is generator
        m_outcome = RR[is_generator[0]]
        assert np.all((is_A_LX_commute(linear_size,kv,LX_vec,LX_sign,LZ_state,bL_state,bR_state,cn_dict)[(LX_sign<=1)])==0), \
            'Stabilizer does not commute with xLog'
        
    elif np.size(is_generator)==0: # A_k not in the table
        s_comm = np.zeros(SS.shape[0]) #whether Nx*Ny stabilizers commute
        d_comm = np.zeros(DD.shape[0]) #whether Nx*Ny-3 destabilizers commute
        for j in range(SS.shape[0]):
            s_comm[j] = is_vertex_commute(linear_size,SS[j,:],bL_state,bR_state,kv,cn_dict)
        for j in range(DD.shape[0]):
            d_comm[j] = is_vertex_commute(linear_size,DD[j,:],bL_state,bR_state,kv,cn_dict)
        s_ncm_ind = np.nonzero(s_comm)[0] #indices of non-commuting SS or DD rows
        d_ncm_ind = np.nonzero(d_comm)[0]

        if np.size(s_ncm_ind)==0: # no non-commuting stabilizers 
            # does A commute with all logical X?
            LX_comm=is_A_LX_commute(linear_size,kv,LX_vec,LX_sign,LZ_state,bL_state,bR_state,cn_dict)
            if np.any(LX_comm == 1):
                LX_sign[LX_comm==1]=2 #no longer eignestate of logical_x[LX_comm==1] so will give logical error if simulating state initialized in logical x
                return 5 #if opt to report logical error whenever it occurs, return 5 without modifying anything
            
            #commute with X logicals ==> deterministic measurement
            buffer_row = A_row.copy() #A_k * \prod g
            buffer_m = 0
            #go throught destabilizers and gather the sign obtained by after multiplying them
            for ind in d_ncm_ind: #the corresponding stabilizers appear in the product that forms A_row
                buffer_row,prod_sign = stab_product(linear_size,buffer_row,SS[ind,:],bL_state,bR_state,cn_dict)
                buffer_m = (buffer_m + RR[ind] + prod_sign)%2 #\prod g, both A and Z elements and the overall sign

            #if A operators are left then deduce them from product of A equality
            if np.count_nonzero(buffer_row[3*Nx*Ny:])>0: #exists any left-over A
                #find color of left-over A operators
                cc = cn_dict['edge_color_arr'][Nx*Ny:2*Nx*Ny] #array corresponding to color of stars
                cc = cc[buffer_row[3*Nx*Ny:]==1] #keep color numbers of left over As
                cc_row = [Nx*Ny-2 , Nx*Ny-1, Nx*Ny-3] #order of rows in which r,g,b relations are stored
                for cc_ind in range(3):
                    if any(cc==cc_ind):
                        # use relation stored in last three rows of SS to remove A operator
                        buffer_row,prod_sign = stab_product(linear_size,buffer_row,SS[cc_row[cc_ind],:],bL_state,bR_state,cn_dict)
                        buffer_m = (buffer_m + RR[cc_row[cc_ind]]+prod_sign)%2 #\prod g, both A and Z elements and the overall sign
            assert np.count_nonzero(buffer_row[3*Nx*Ny:])==0 ,'Error: last three rows are insufficient to cancel A'
            #now buffer_row should only have Z decorations, and buffer_m is s^\prime
            if np.count_nonzero(buffer_row)==0: # if no z-operators left
                m_outcome = buffer_m
            else: # closed loop of Z = prod_B is left, so find the sign of this prod_B check appendix for details
                m_outcome = (parity_of_B_inside(linear_size,buffer_row[:3*Nx*Ny],bL_state,bR_state,LZ_state,LX_sign,cn_dict) + buffer_m)%2

        else: # some anticommute ==> probabilistic measurement, 0 or 1 outcome with equal prob
            # to project probabilistically and update the stabilizers accordingly
            # multiply non-commuting D,S by first non-commuting stabilizers
            # replace corresponding D by S
            # similarly update non-commuting logical X --> S*logical_x
            assert np.count_nonzero(s_ncm_ind<(Nx*Ny-3))>0, 'Error: non-commuting stabilizers'

            for ind in d_ncm_ind: #update destabilizers, Eq C14
                DD[ind,:],prod_sign = stab_product(linear_size,DD[ind,:],SS[s_ncm_ind[0],:],bL_state,bR_state,cn_dict)

            LX_comm = is_A_LX_commute(linear_size,kv,LX_vec,LX_sign,LZ_state,bL_state,bR_state,cn_dict)
            LX_update_ind = np.nonzero(LX_comm==1)[0]
            for ind in LX_update_ind: #update LX stabilizer
                LX_vec[ind,:],prod_sign = stab_product(linear_size,LX_vec[ind,:],SS[s_ncm_ind[0],:],bL_state,bR_state,cn_dict)
                LX_sign[ind] = (RR[s_ncm_ind[0]] + LX_sign[ind] + prod_sign)%2

            for ind in s_ncm_ind[1:]: #update stabilizers
                SS[ind,:],prod_sign = stab_product(linear_size,SS[ind,:],SS[s_ncm_ind[0],:],bL_state,bR_state,cn_dict)
                RR[ind] = (RR[s_ncm_ind[0]] + RR[ind] + prod_sign)%2
                
            DD[s_ncm_ind[0],:] = SS[s_ncm_ind[0],:].copy()
            SS[s_ncm_ind[0],:] = A_row.copy()

            #generate random outcome
            m_outcome = rnd.randint(0,1)
            RR[s_ncm_ind[0]] = m_outcome
    else:
        m_outcome = None
        print('Error: The stabilizer rows are duplicated',np.size(is_generator))
    return m_outcome

def is_vertex_commute(linear_size,s_row,bL_state,bR_state,k,cn_dict):
    '''
    Does opeator A_{k} commute with operator row
    Input:
        linear_size
        s_row : stabilizer matrix row (4*Nx*Ny,) shaped array of 0 and 1, i.e. [1D array]
        bL_state : current Bl vector 
        bR_state : current Br vector
        k : vertex index of measurement operator
        vv : array of corner vertices (output of find_hexagon_corners)
    Output:
        out : out = 0 if commute and out = 1 if anti-commutes
    '''    
    Nx,Ny=linear_size
    assert s_row.shape==(4*Nx*Ny,)
    outer_edges = cn_dict['star_stored_edges'][k,:] #edge numbers of the X edges of A_k
    out = np.count_nonzero(s_row[outer_edges])%2 #commutation relation with Z dressings

    vv = cn_dict['hex_corners'][k,:] #6 star neighbors of A_k, with whom A_k commutes up to 2 triangles
    if np.count_nonzero(s_row[3*Nx*Ny+vv])>0: #if more than one of the 6 neighbors exist in s_row
        #record states of vertex B operators at kv and on the 6 neighbors
        corner_b = np.array([bR_state[vv[0]],bL_state[vv[1]],bR_state[vv[2]],
                            bL_state[vv[3]],bR_state[vv[4]],bL_state[vv[5]]])
        central_b = np.array([bL_state[k],bR_state[k],bL_state[k],
                            bR_state[k],bL_state[k],bR_state[k]])
        #commutation relation with 6 neighboring stars
        for i in range(6):
            if s_row[3*Nx*Ny + vv[i]]==1: #if the neighbor is in s_row
                out = (out + central_b[i] + corner_b[i])%2        
    return out

def stab_product(linear_size,f_row,g_row,bL_state,bR_state,cn_dict): 
    '''
    Input 
        f_row ,g_row : two stabilizer tableau rows
    Returns:
        f_row = f_row*g_row
        foo = sign(f_row), additional sign on top of RR

        modifies: f_row to be f_row = f_row*g_row, g_row unchanged
    '''
    Nx,Ny = linear_size
    g_occ=np.nonzero(g_row)[0] #index of nonzero elements in g_row
    foo=0
    
    for j in g_occ[g_occ<3*Nx*Ny] : # move g[z] through f[A] and annihilate with f[z]
        #where Z in g may anticommute with A in f
        ncm_star_index = cn_dict['star_syndromes'][j,:]
        foo = (foo + f_row[3*Nx*Ny+ncm_star_index[0]]+f_row[3*Nx*Ny+ncm_star_index[1]])%2 #commute through f[A]
        f_row[j] = (f_row[j]+1)%2 #annihilate with f[z]
        
    for j in g_occ[g_occ>=3*Nx*Ny]: #move g[A_k] upto f[A_k]
        f_mask = f_row*(np.arange(4*Nx*Ny)>j) #only look at f[A_k>j]
        kv=j-3*Nx*Ny
        foo = (foo + is_vertex_commute(linear_size,f_mask,bL_state,bR_state,kv,cn_dict))%2 #commute A_j through f[A_k>j]
        f_row[j] = (f_row[j]+g_row[j])%2 #annihilate with f[A_j]
    return f_row,foo

def is_A_LX_commute(linear_size,kv,LX_vec,LX_sign,LZ_state,bL_state,bR_state,cn_dict): 
    '''
    Determine if A_{kv} commutes with the logical X operator

    Returns comm = (6,) array. comm[j] = 0 if LX_vec[j] commutes with A_{kv} and 1 if anticommutes with A_{kv}
            default to comm=2 if particular Lx is not the current stabilizer because Lx_sign >1
    '''
    assert kv<linear_size[0]*linear_size[1]
    assert (bL_state[kv]==0 and bR_state[kv]==0), ('B != +1 where A is measured')

    comm = np.zeros(np.shape(LX_vec)[0]) #6 elements, corresponding to each LX
    for row in range(LX_vec.shape[0]): #loop through 6 LXs
        if LX_sign[row]<=1: # check commutation of A with (Z,A) decorations in logical X
            comm[row] = is_vertex_commute(linear_size,LX_vec[row,:],bL_state,bR_state,kv,cn_dict)
            
            try: # if A is at the bottom or rightmost column then additional logical Z appears in commutation (see construct_bottom_dictionary())
                comm[row] = (comm[row]+ LZ_state[cn_dict['bottom_dict'][(row,kv)]])%2
            except:
                pass
        else:
            comm[row] = 2

    assert np.all(comm[LX_sign[row]<=1]<=1)
    assert np.array_equal(LX_sign==2 ,comm==2) #does not activate or deactivate LX
    return comm

def parity_of_B_inside(linear_size,loop_z_edges,bL_state,bR_state,LZ_state,LX_sign,cn_dict): 
    '''
    Parity of B operator violations that participate to construct of loop_z_edges on dual lattice
    Modified to include the possibility where loop_z_edges is not contractible and hence involves product of lz_state

    if strings cross odd with loop_z odd number of times while pairing B defects, then the prod_B_inside = -1

    if z_loop = zlog+B then find which z-log is involved and record its sign, then remove it create a bounded region
    Returns: parity int 0 or 1
    '''
    assert(np.shape(loop_z_edges)==(3*linear_size[0]*linear_size[1],))
    # check that z_edges form a closed loop on the dual lattice
    # closed Z loops, contractible or non-contractible, must commute with all stars
    num_defects = np.count_nonzero((cn_dict['HH_star']@loop_z_edges)%2)
    assert num_defects==0 ,'z_edges do not form a loop'

    #is the loop non-contractible, non-contractible if anti-commute with LX
    x_flipped = np.nonzero((cn_dict['HH_log_x']@loop_z_edges)%2)[0]
    sign = 0
    for ind in x_flipped:
        # add additional z-logical to create a closed loop that bounds a region
        assert LX_sign[ind]==2 ,'Using Z to determine outcome but state is X eigenstate'
        loop_z_edges = (loop_z_edges + cn_dict['HH_log_z'][(ind+3)%6,:])%2 #remove the logical
        sign  = (sign + LZ_state[(ind+3)%6])%2 #the sign of the LZ indicates whether an odd number of X strings passed through
    for cc in range(3):
        syndrome = colored_syndrome(cc,bL_state,bR_state,cn_dict)   
        prediction = cn_dict['mgraph'][cc].decode(syndrome) #dot product with loop_z_edges to determine overlap
        sign = (sign + (loop_z_edges@prediction)%2)%2

    return sign%2

def colored_syndrome(cc,bL_state,bR_state,cn_dict):
    '''
    Find syndromes of color cc given list of bL and bR syndromes
    Input:
        cc = scalar color 0,1 or 2
    Returns:
        B syndrome = (Nx*Ny,) shaped array 
    '''
    return ((cn_dict['bL_vertex_color_arr']==cc)*bL_state) + ((cn_dict['bR_vertex_color_arr']==cc)*bR_state)

def correct_e_anyons(linear_size,bL_state,bR_state,SS,DD,RR,LZ_state,LX_vec,LX_sign,cn_dict):
    '''
    Perform measurement and MWPM of e-anyons after all m-anyons are corrected
    Raise exception if m-anyons are not corrected

    Parameters
    ----------
    linear_size : 
        (Nx,Ny)
    bL_state : 
        state of bL 
    bR_state : 
        state bR
    SS : 
        stabilizer matrix of A-type stabilizers
    DD : 
        destabilizer matrix
    RR : 
        sign of SS rows
    LZ_state : 
        current logical z state
    LX_vec : 
        operator matrix for logical-x
    LX_sign : 
        signs of LX_vec
    cn_dict : 
        precomputed dictionary of lattice structure

    Returns
    -------
        0 if measurement and correction is completed, 1 if e-anyons cannot be corrected due to odd number of anyons for any color
    '''
    Nx,Ny = linear_size
    #raise an error if m anyons are not fully corrected
    assert np.array_equal(bL_state,np.zeros(Nx*Ny))
    assert np.array_equal(bR_state,np.zeros(Nx*Ny))

    a_syndrome = np.zeros(Nx*Ny)
    #detect A defects by measurement
    for kv in range(Nx*Ny):
        a_syndrome[kv]=measure_A(kv,linear_size,SS,DD,RR,bL_state,bR_state,LZ_state,LX_vec,LX_sign,cn_dict)
    #apply z to remove A defects
    try:
        z_correction_locations = np.nonzero(cn_dict['mgraph'][3].decode(a_syndrome))[0]
        for z_edge in z_correction_locations:
            apply_Z(z_edge,linear_size,SS,RR,LX_vec,LX_sign,cn_dict)
        return 0 #return 0 if the correction is completed
    except ValueError: # No matching found, odd number of e-anyon of any color
        return 5 #return 1 if there are odd number of e-anyons for each color