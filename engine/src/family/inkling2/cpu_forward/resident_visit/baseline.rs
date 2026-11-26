// The original resident path; the decode-glue switch leaves this path unchanged.
use super::*;

impl Execution<'_> {
    pub(super) fn enqueue_baseline(
        &self,
        cuda: &GgufCuda,
        position: usize,
    ) -> Result<(), EngError> {
        let l = self.layout;
        let shape = self.plan.shape;
        let (depth, bank) = ((l.depth, 0), (l.bank, shape.bank_first));
        for (query, out) in self
            .weights
            .bank_queries
            .into_iter()
            .flatten()
            .zip(l.messages)
        {
            self.mix(bank, shape.bank_blocks, query, out)?;
        }
        let input = match self.weights.attention_query {
            Some(query) => {
                self.mix(depth, shape.blocks, query, l.attention_input)?;
                l.attention_input
            }
            None => l.depth.row(shape.widths.hidden),
        };
        self.attention(cuda, position, input)?;
        let blocks = self.advance_depth_baseline()?;
        self.mix(depth, blocks, self.weights.mlp_query, l.mlp_input)?;
        self.mlp_baseline(cuda)?;
        self.add(l.mlp_conv)
    }

    pub(super) fn advance_depth_baseline(&self) -> Result<usize, EngError> {
        let shape = self.plan.shape;
        let l = self.layout;
        if !shape.seal {
            self.add(l.attention_out)?;
            return Ok(shape.blocks);
        }
        self.context.inkling_stream_seal(
            self.stream,
            SealCall {
                state: self.row(l.depth)?,
                next: self.row(l.attention_out)?,
                health: self.row(l.glue_health)?,
            },
            StreamShape {
                blocks: shape.blocks,
                width: shape.widths.hidden,
                eps: MIX_EPS,
            },
        )?;
        Ok(shape.blocks + 1)
    }

    pub(super) fn add(&self, addend: Region) -> Result<(), EngError> {
        let width = self.plan.shape.widths.hidden;
        self.context.inkling_stream_add(
            self.stream,
            crate::kernels::cuda::resident::inkling_stream::AddCall {
                state: self.row(self.layout.depth.row(width))?,
                row: self.row(addend)?,
                health: self.row(self.layout.glue_health)?,
            },
            width,
        )
    }

    pub(super) fn mlp_baseline(&self, cuda: &GgufCuda) -> Result<(), EngError> {
        let l = self.layout;
        let [gamma, bias, conv] = self.weights.mlp;
        let controls = &self.plan.controls;
        let request = ResidentMlpRequest {
            layer: self.plan.visit.weight_layer as u32,
            caps: controls.caps,
            scale: controls.scale,
            dtypes: controls.dtypes,
            controller: controls.names.each_ref().map(String::as_str),
            rank: controls.rank,
            eps: self.plan.math.eps,
            input: self.row(l.mlp_input)?,
            message: self.row(l.messages[3])?,
            gamma: self.weights.row(gamma)?,
            bias: self.weights.row(bias)?,
            out: self.row(l.mlp_raw)?,
            health: self.row(l.mlp_health)?,
            gate_health: self.row(l.glue_health)?,
            status: self.row(l.mlp_status)?,
            pre_normed: None,
        };
        if self.plan.shape.banked {
            cuda.enqueue_controlled_mlp(self.stream, request)?;
        } else {
            cuda.enqueue_plain_mlp(self.stream, request)?;
        }
        let shape = self.plan.shape.widths;
        self.context.inkling_conv(
            self.stream,
            ConvCall {
                input: self.row(l.mlp_raw)?,
                kernel: self.weights.row(conv)?,
                history: self.row(l.histories[3])?,
                out: self.row(l.mlp_conv)?,
                state: None,
                health: self.row(l.glue_health)?,
            },
            (shape.hidden, shape.taps),
        )
    }
}
