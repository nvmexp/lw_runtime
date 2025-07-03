%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

test_types = {'double', 'single', 'uint32', 'int32', 'uint16', 'int16', 'uint8', 'int8'};

for idx = 1:numel(test_types)
    t = test_types{idx};
    fname = strcat('magic_', t, '.h5');
    A = cast(magic(16), t);
    h5File  = H5F.create(fname, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
    hdf5_write_lw(h5File, 'A', A);
    H5F.close(h5File);

    check = hdf5_load_lw(fname);
    success_count = 0;
    if size(A) ~= size(check.A)
        fprintf('Size error for %s\n', fname);
        size(A)
        size(check.A)
    else
        success_count = success_count + 1;
    end
    if any(A(:) ~= check.A(:))
        fprintf('Data mismatch for %s\n', fname);
        A
        check.A
    else
        success_count = success_count + 1;
    end
    if 2 == success_count
        fprintf('%s: SUCCESS\n', fname);
        delete(fname);
    end
end
