import numpy as np
from muDIC.solver.reference import find_covered_pixel_blocks
from muDIC.elements.b_splines import BSplineSurface



def initial_conds_from_analysis(org_mesh, target_mesh, dic_results):
    """
   Mesh translator maps the nodal position history obtained with org_mesh to the corresponding
   nodal position history for target_mesh.
   The intended use of this function is to allow for a result set to be used as initial conditions for another analysis.
   TODO: Write tests
   :param org_mesh: mesh instance
   :param target_mesh: mesh instance
   :param dic_results: dic_results instance
   :return: xnodesT,ynodeT corresponding to target_mesh
   """


    # TODO: Add Q4 support

    if not isinstance(org_mesh.element_def,BSplineSurface):
        raise NotImplementedError("Only B-spline elements are supported as original mesh")

    print("The mesh translator is in Beta and may yield invalid results!")

    # Find the element coordinates for the target mesh nodes
    es, ns, xs, ys = find_covered_pixel_blocks(org_mesh.xnodes.flatten(),
                                               org_mesh.ynodes.flatten(), org_mesh.element_def,
                                               xs=target_mesh.xnodes.flatten(), ys=target_mesh.ynodes.flatten(),
                                               keep_all=True)

    # Some coordinates can be negative and very small.
    es = np.round(np.array(es).flatten(),decimals=4)
    ns = np.round(np.array(ns).flatten(),decimals=4)

    # Find the nodal positions of the target mesh based on the deformation of the original mesh
    node_x = np.dot(org_mesh.element_def.Nn(es, ns), dic_results.xnodesT[:, :])
    node_y = np.dot(org_mesh.element_def.Nn(es, ns), dic_results.ynodesT[:, :])
    return node_x, node_y
