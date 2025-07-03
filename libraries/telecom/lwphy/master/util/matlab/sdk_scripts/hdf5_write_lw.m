 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function hdf5_write_lw(hdf5loc, name, A)

  % Determine the element type
  switch(class(A))
    case 'single'
      H5TypeString = 'H5T_NATIVE_FLOAT';
    case 'double'
      H5TypeString = 'H5T_NATIVE_DOUBLE';
    case 'uint8'
      H5TypeString = 'H5T_NATIVE_UINT8';
    case 'uint16'
      H5TypeString = 'H5T_NATIVE_UINT16';
    case 'uint32'
      H5TypeString = 'H5T_NATIVE_UINT32';
    case 'uint64'
      H5TypeString = 'H5T_NATIVE_UINT64';
    case 'int8'
      H5TypeString = 'H5T_NATIVE_INT8';
    case 'int16'
      H5TypeString = 'H5T_NATIVE_INT16';
    case 'int32'
      H5TypeString = 'H5T_NATIVE_INT32';
    case 'int64'
      H5TypeString = 'H5T_NATIVE_INT64';
    otherwise
      error('Unexpected element class (%s)', class(A))
  end
    
  elemType       = H5T.copy(H5TypeString);
  if isreal(A)
    Awrite = A;
    fileType = elemType;
  else
    % Create an HDF5 complex data type
    compoundSizes   = [H5T.get_size(elemType) H5T.get_size(elemType)];
    compoundOffsets = [0 lwmsum(compoundSizes(1:end-1))];
    complexType     = H5T.create('H5T_COMPOUND', sum(compoundSizes));
    H5T.insert(complexType, 're', compoundOffsets(1), elemType);
    H5T.insert(complexType, 'im', compoundOffsets(2), elemType);
    % Copy real and imaginary data to a local struct with two fields
    Awrite.re = real(A);
    Awrite.im = imag(A);
    fileType = complexType;
  end
  
  % Create a dataspace. MATLAB is column major, whereas HDF5 is
  % row major, so we do a fliplr on the dimensions
  % Setting maximum size to [] indicates that the maximum size is
  % current size.
  arrayDataspace = H5S.create_simple(ndims(A), fliplr(size(A)), []);

  % Create a dataset and write it to the file
  arrayDataset = H5D.create(hdf5loc, name, fileType, arrayDataspace, 'H5P_DEFAULT');
  H5D.write(arrayDataset, fileType, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', Awrite);

  % Cleanup
  H5D.close(arrayDataset);
  H5S.close(arrayDataspace);
  H5T.close(fileType);
end



