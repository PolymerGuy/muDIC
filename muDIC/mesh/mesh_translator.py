def mesh_translator(org_mesh, target_mesh, dic_results):
   """
   Mesh translator maps the nodal position history obtained with org_mesh to the corresponding
   nodal position history for target_mesh.
   The intended use of this function is to allow for a result set to be used as initial conditions for another analysis.
   TODO: This implementation currently only supports org_mesh being a single spline element.
   TODO: Write tests
   :param org_mesh: mesh instance
   :param target_mesh: mesh instance
   :param dic_results: dic_results instance
   :return: xnodesT,ynodeT corresponding to target_mesh
   """
   if org_mesh.ele.shape[1] is not 1:
       raise IOError("The origin mesh has to be a single element mesh!")
   if org_mesh.ynodes.shape[0] is not dic_results.xnodesT.shape[0]:
       raise IOError("The node-positions does not correspond to the original mesh")
   # We need the element coordinates to the nodes
   # We do this by using the original and target mesh in its un-deformed state
   es, ns, _, _ = find_element_coordinates(org_mesh.xnodes[org_mesh.ele[:, 0]],
                                           org_mesh.ynodes[org_mesh.ele[:, 0]], org_mesh.element_def,
                                           Xx=target_mesh.xnodes.flatten(),
                                           Yy=target_mesh.ynodes.flatten())
   node_x = np.dot(target_mesh.element_def.Nn(es, ns), dic_results.xnodesT[org_mesh.ele[:, 0], :])
   node_y = np.dot(target_mesh.element_def.Nn(es, ns), dic_results.ynodesT[org_mesh.ele[:, 0], :])
   return node_x, node_y