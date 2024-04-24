import gradio
import torch
import modules


def _rejection(u, v):
    uv = torch.mean(u * v, dim=list(range(1, v.ndim)), keepdim=True)
    vv = torch.mean(v ** 2, dim=list(range(1, v.ndim)), keepdim=True) + 1.0e-05
    return u - (uv / vv) * v


class PredictionBoostForForge(modules.scripts.Script):
    sorting_priority = 115

    def title(self):
        return "Prediction Boost"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gradio.Accordion(open=False, label=self.title()):
            enabled = gradio.Checkbox(label='Enabled', value=False)
            boost_scale = gradio.Slider(label='Boost Scale', minimum=0.0, maximum=1.0, step=0.01, value=0.05)
        return enabled, boost_scale

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, boost_scale = script_args
        if not enabled:
            return

        def sampler_post_cfg_function(args):
            input = args["input"]
            cond_denoised = args["cond_denoised"]
            denoised = args["denoised"]
            return denoised + boost_scale * _rejection(input, input - cond_denoised)

        model = p.sd_model.forge_objects.unet.clone()
        model.set_model_sampler_post_cfg_function(sampler_post_cfg_function)
        p.sd_model.forge_objects.unet = model

        p.extra_generation_params.update(dict(
            predictionboost_enabled=enabled,
            predictionboost_scale=boost_scale,
        ))
