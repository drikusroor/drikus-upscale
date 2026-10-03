"""Stand-in for BiRefNet's config.py, pinned to the BiRefNet_lite release.

Upstream's Config() reads the training file system and shell scripts at
construction time; the model code only needs the architecture switches below.
Each value matches upstream config.py at the vendored commit with
`bb = 'swin_v1_t'` -- the settings the
BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth checkpoint was trained with.
tools/bgremove/export.py loads that checkpoint with strict=True, so a wrong
switch here fails loudly instead of exporting a different network.
"""


class Config():
    def __init__(self) -> None:
        self.batch_size = 8   # > 1 means the decoder carries BatchNorm layers
        self.task = 'General'
        self.size = (1024, 1024)

        self.ms_supervision = True
        self.out_ref = self.ms_supervision and True
        self.dec_ipt = True
        self.dec_ipt_split = True
        self.cxt_num = 3
        self.mul_scl_ipt = 'cat'
        self.dec_att = 'ASPPDeformable'
        self.squeeze_block = 'BasicDecBlk_x1'
        self.dec_blk = 'BasicDecBlk'

        self.bb = 'swin_v1_t'
        self.freeze_bb = False
        self.lateral_channels_in_collection = [768, 384, 192, 96]
        if self.mul_scl_ipt == 'cat':
            self.lateral_channels_in_collection = [c * 2 for c in self.lateral_channels_in_collection]
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:] if self.cxt_num else []

        self.lat_blk = 'BasicLatBlk'
        self.dec_channels_inter = 'fixed'
        self.auxiliary_classification = False
        self.model = 'BiRefNet'
        self.SDPA_enabled = False   # plain matmul attention: traces to ops every EP supports
        self.precisionHigh = True
