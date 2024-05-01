import gradio
import torch
import modules


def _rejection(u, v):
    uv = torch.mean(u * v, dim=list(range(1, v.ndim)), keepdim=True)
    vv = torch.mean(v ** 2, dim=list(range(1, v.ndim)), keepdim=True) + 1.0e-05
    return u - (uv / vv) * v


def _normalize(u, v):
    uu = torch.mean(u ** 2, dim=list(range(1, u.ndim)), keepdim=True) + 1.0e-05
    vv = torch.mean(v ** 2, dim=list(range(1, v.ndim)), keepdim=True)
    return ((torch.relu(1 - vv) / uu) ** 0.5) * u


class PredictionBoostForForge(modules.scripts.Script):
    sorting_priority = 115

    def title(self):
        return "Prediction Boost"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with modules.ui_components.InputAccordion(False, label=self.title()) as enabled:
            do_normalize = gradio.Checkbox(label="Normalize", value=True)
            boost_scale = gradio.Slider(label='Boost Scale', minimum=0.0, maximum=1.0, step=0.01, value=0.10)
        return enabled, do_normalize, boost_scale

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, do_normalize, boost_scale = script_args

        if not enabled:
            return

        def sampler_post_cfg_function(args):
            sigma = args["sigma"]
            input = args["input"]
            denoised = args["denoised"]
            cond_denoised = args["cond_denoised"]

            noise_pred_cond = (input - cond_denoised) / sigma
            boost_denoised = _rejection(input, noise_pred_cond)
            if do_normalize:
                boost_denoised = _normalize(boost_denoised, noise_pred_cond) * sigma
            return denoised + boost_scale * boost_denoised

        model = p.sd_model.forge_objects.unet.clone()
        model.set_model_sampler_post_cfg_function(sampler_post_cfg_function)
        p.sd_model.forge_objects.unet = model

        p.extra_generation_params.update(dict(
            predictionboost_normalize=do_normalize,
            predictionboost_scale=boost_scale,
        ))
