import Metashape
root = '/path/to/your/project/'
doc = Metashape.app.document
doc.open(root + 'project.psz')
for chunk in doc.chunks:
    # Align Photos
    for frame in chunk.frames:
        frame.matchPhotos(
            downscale=0,
            generic_preselection=False,
            reference_preselection=True,
            reference_preselection_mode=Metashape.ReferencePreselectionSource,
            filter_mask=False,
            mask_tiepoints=True,
            keypoint_limit=0,
            tiepoint_limit=0,
            keep_keypoints=True,
            guided_matching=False,
            reset_matches=False,
            # subdivide_task=False
        )
    chunk.alignCameras(
        min_image=2,
        adaptive_fitting=True,
        reset_alignment=False,
        # subdivide_task=False
    )
    chunk.exportCameras(
        path=root + 'cams.xml',
        format=Metashape.CamerasFormatXML,
        save_points=False,
        use_initial_calibration=True
    )
    for frame in chunk.frames:
        # Build Mesh
        frame.buildDepthMaps(
            downscale=1,
            filter_mode=Metashape.MildFiltering,
            reuse_depth=False,
            max_neighbors=-1,
            # subdivide_task=False
        )
        frame.buildModel(
            surface_type=Metashape.Arbitrary,
            interpolation=Metashape.EnabledInterpolation,
            face_count=Metashape.HighFaceCount,
            face_count_custom=200000,
            source_data=Metashape.DepthMapsData,
            vertex_colors=False,
            vertex_confidence=False,
            volumetric_masks=False,
            keep_depth=False,
            trimming_radius=10,
            # subdivide_task=False
        )
        # # Build Texture
        # frame.buildUV(
        #     mapping_mode=Metashape.GenericMapping,
        #     page_count=1,
        #     texture_size=8192
        # )
        # frame.buildTexture(
        #     blending_mode=Metashape.MosaicBlending,
        #     texture_size=8192,
        #     fill_holes=True,
        #     ghosting_filter=True,
        #     texture_type=Metashape.DiffuseMap,
        #     transfer_texture=True
        # )
        frame.exportModel(
            path=root + '1.obj',
            texture_format=Metashape.ImageFormatPNG,
            save_texture=True,
            save_uv=True,
            save_normals=True,
            save_colors=True,
            save_confidence=False,
            save_cameras=False,
            save_markers=False,
            save_comment=False
        )
doc.save()