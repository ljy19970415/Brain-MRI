RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3
import numpy as np
from .augmentation.augmentations.utils import resize_segmentation

def determine_whether_to_use_mask_for_norm(self):
    # only use the nonzero mask for normalization of the cropping based on it resulted in a decrease in
    # image size (this is an indication that the data is something like brats/isles and then we want to
    # normalize in the brain region only)
    modalities = self.dataset_properties['modalities']
    num_modalities = len(list(modalities.keys()))
    use_nonzero_mask_for_norm = OrderedDict()

    for i in range(num_modalities):
        if "CT" in modalities[i]:
            use_nonzero_mask_for_norm[i] = False
        else:
            all_size_reductions = []
            for k in self.dataset_properties['size_reductions'].keys():
                all_size_reductions.append(self.dataset_properties['size_reductions'][k])

            if np.median(all_size_reductions) < 3 / 4.:
                print("using nonzero mask for normalization")
                use_nonzero_mask_for_norm[i] = True
            else:
                print("not using nonzero mask for normalization")
                use_nonzero_mask_for_norm[i] = False

    for c in self.list_of_cropped_npz_files:
        case_identifier = get_case_identifier_from_npz(c)
        properties = self.load_properties_of_cropped(case_identifier)
        properties['use_nonzero_mask_for_norm'] = use_nonzero_mask_for_norm
        self.save_properties_of_cropped(case_identifier, properties)
    use_nonzero_mask_for_normalization = use_nonzero_mask_for_norm
    return use_nonzero_mask_for_normalization

def get_target_spacing(self):
    spacings = self.dataset_properties['all_spacings']

    # target = np.median(np.vstack(spacings), 0)
    # if target spacing is very anisotropic we may want to not downsample the axis with the worst spacing
    # uncomment after mystery task submission
    """worst_spacing_axis = np.argmax(target)
    if max(target) > (2.5 * min(target)):
        spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
        target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 5)
        target[worst_spacing_axis] = target_spacing_of_that_axis"""

    target = np.percentile(np.vstack(spacings), self.target_spacing_percentile, 0)
    return target

def plan_experiment(self):
    use_nonzero_mask_for_normalization = self.determine_whether_to_use_mask_for_norm()
    print("Are we using the nonzero mask for normalization?", use_nonzero_mask_for_normalization)
    spacings = self.dataset_properties['all_spacings']
    sizes = self.dataset_properties['all_sizes']

    all_classes = self.dataset_properties['all_classes']
    modalities = self.dataset_properties['modalities']
    num_modalities = len(list(modalities.keys()))

    target_spacing = self.get_target_spacing()
    new_shapes = [np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)]

    max_spacing_axis = np.argmax(target_spacing)
    remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
    self.transpose_forward = [max_spacing_axis] + remaining_axes
    self.transpose_backward = [np.argwhere(np.array(self.transpose_forward) == i)[0][0] for i in range(3)]

    # we base our calculations on the median shape of the datasets
    median_shape = np.median(np.vstack(new_shapes), 0)
    print("the median shape of the dataset is ", median_shape)

    max_shape = np.max(np.vstack(new_shapes), 0)
    print("the max shape in the dataset is ", max_shape)
    min_shape = np.min(np.vstack(new_shapes), 0)
    print("the min shape in the dataset is ", min_shape)

    print("we don't want feature maps smaller than ", self.unet_featuremap_min_edge_length, " in the bottleneck")

    # how many stages will the image pyramid have?
    self.plans_per_stage = list()

    target_spacing_transposed = np.array(target_spacing)[self.transpose_forward]
    median_shape_transposed = np.array(median_shape)[self.transpose_forward]


def preprocess():
    d, s, properties = preprocess_test_case(input_files,target_spacing_transposed)

def preprocess_test_case(data_files, target_spacing, seg_file=None, force_separate_z=None):
    # crop the zero part in data, and keep the nonzero part
    data, seg, properties = ImageCropper.crop_from_list_of_files(data_files, seg_file)

    data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
    seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))

    data, seg, properties = self.resample_and_normalize(data, target_spacing, properties, seg,
                                                        force_separate_z=force_separate_z)
    return data.astype(np.float32), seg, properties

def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
    """
    data and seg must already have been transposed by transpose_forward. properties are the un-transposed values
    (spacing etc)
    :param data:
    :param target_spacing:
    :param properties:
    :param seg:
    :param force_separate_z:
    :return:
    """

    # target_spacing is already transposed, properties["original_spacing"] is not so we need to transpose it!
    # data, seg are already transposed. Double check this using the properties
    original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
    before = {
        'spacing': properties["original_spacing"],
        'spacing_transposed': original_spacing_transposed,
        'data.shape (data is transposed)': data.shape
    }

    # remove nans
    data[np.isnan(data)] = 0

    data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing,
                                    self.resample_order_data, self.resample_order_seg,
                                    force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
                                    separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)
    after = {
        'spacing': target_spacing,
        'data.shape (data is resampled)': data.shape
    }
    print("before:", before, "\nafter: ", after, "\n")

    if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
        seg[seg < -1] = 0

    properties["size_after_resampling"] = data[0].shape
    properties["spacing_after_resampling"] = target_spacing
    use_nonzero_mask = self.use_nonzero_mask

    assert len(self.normalization_scheme_per_modality) == len(data), "self.normalization_scheme_per_modality " \
                                                                        "must have as many entries as data has " \
                                                                        "modalities"
    assert len(self.use_nonzero_mask) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                    " has modalities"

    for c in range(len(data)):
        scheme = self.normalization_scheme_per_modality[c]
        if scheme == "CT":
            # clip to lb and ub from train data foreground and use foreground mn and sd from training data
            assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
            mean_intensity = self.intensityproperties[c]['mean']
            std_intensity = self.intensityproperties[c]['sd']
            lower_bound = self.intensityproperties[c]['percentile_00_5']
            upper_bound = self.intensityproperties[c]['percentile_99_5']
            data[c] = np.clip(data[c], lower_bound, upper_bound)
            data[c] = (data[c] - mean_intensity) / std_intensity
            if use_nonzero_mask[c]:
                data[c][seg[-1] < 0] = 0
        elif scheme == "CT2":
            # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
            assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
            lower_bound = self.intensityproperties[c]['percentile_00_5']
            upper_bound = self.intensityproperties[c]['percentile_99_5']
            mask = (data[c] > lower_bound) & (data[c] < upper_bound)
            data[c] = np.clip(data[c], lower_bound, upper_bound)
            mn = data[c][mask].mean()
            sd = data[c][mask].std()
            data[c] = (data[c] - mn) / sd
            if use_nonzero_mask[c]:
                data[c][seg[-1] < 0] = 0
        elif scheme == 'noNorm':
            print('no intensity normalization')
            pass
        else:
            if use_nonzero_mask[c]:
                mask = seg[-1] >= 0
                data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                data[c][mask == 0] = 0
            else:
                mn = data[c].mean()
                std = data[c].std()
                # print(data[c].shape, data[c].dtype, mn, std)
                data[c] = (data[c] - mn) / (std + 1e-8)
    return data, seg, properties


def resample_patient(data, seg, original_spacing, target_spacing, order_data=3, order_seg=0, force_separate_z=False,
                     order_z_data=0, order_z_seg=0,
                     separate_z_anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    """
    :param data:
    :param seg:
    :param original_spacing:
    :param target_spacing:
    :param order_data:
    :param order_seg:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately
    :param order_z_seg: only applies if do_separate_z is True
    :param order_z_data: only applies if do_separate_z is True
    :param separate_z_anisotropy_threshold: if max_spacing > separate_z_anisotropy_threshold * min_spacing (per axis)
    then resample along lowres axis with order_z_data/order_z_seg instead of order_data/order_seg

    :return:
    """
    assert not ((data is None) and (seg is None))
    if data is not None:
        assert len(data.shape) == 4, "data must be c x y z"
    if seg is not None:
        assert len(seg.shape) == 4, "seg must be c x y z"

    if data is not None:
        shape = np.array(data[0].shape)
    else:
        shape = np.array(seg[0].shape)
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        data_reshaped = resample_data_or_seg(data, new_shape, False, axis, order_data, do_separate_z,
                                             order_z=order_z_data)
    else:
        data_reshaped = None
    if seg is not None:
        seg_reshaped = resample_data_or_seg(seg, new_shape, True, axis, order_seg, do_separate_z, order_z=order_z_seg)
    else:
        seg_reshaped = None
    return data_reshaped, seg_reshaped


def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, order_z=0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    # for seg order = 1 order_z = 0
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == len(data.shape) - 1
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                                   mode='nearest')[None].astype(dtype_data))
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None].astype(dtype_data))
                else:
                    reshaped_final_data.append(reshaped_data[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data