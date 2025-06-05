"""
Utilities related to sampling-based optimization with BO.

@yjy0625
"""
import os
import numpy as np
from tqdm import tqdm
from skopt import Optimizer
from skopt.space import Real
import cma

import plotly.graph_objects as go


def normalize(x, bounds):
    """
    Normalize data to [0, 1] range.
    
    Parameters:
    - x (np.ndarray): Input data.
    - bounds (list of tuples): [(min_x, max_x), (min_y, max_y), ...].
    
    Returns:
    - normalized_x (np.ndarray): Normalized data.
    """
    return (x - np.array([b[0] for b in bounds])) / np.array([b[1] - b[0] for b in bounds])


def denormalize(x, bounds):
    """
    Denormalize data from [0, 1] range back to original bounds.
    
    Parameters:
    - x (np.ndarray): Normalized data.
    - bounds (list of tuples): [(min_x, max_x), (min_y, max_y), ...].
    
    Returns:
    - original_x (np.ndarray): Data in original scale.
    """
    return x * np.array([b[1] - b[0] for b in bounds]) + np.array([b[0] for b in bounds])


def optimize_pose_batch(F, bounds, algorithm='bayesian', n_iterations=100, batch_size=10, 
                        normalize_data=True, track_info=True, initial_samples=None, seed=0,
                        print_fn=lambda a: a, **kwargs):
    """
    Optimize the robot base pose (x, y, a) using batch evaluation with optional data normalization.
    
    Parameters:
    - F (function): The function to optimize. Takes a numpy array of poses and returns a list of scores.
    - bounds (list of tuples): The bounds for the optimization, [(min_x, max_x), (min_y, max_y), (min_a, max_a)].
    - algorithm (str): The optimization algorithm to use, either 'bayesian' or 'cmaes'.
    - n_iterations (int): Number of iterations for the optimization.
    - batch_size (int): Number of samples to evaluate in each batch.
    - normalize_data (bool): Whether to normalize data to [0, 1] internally.
    - track_info (bool): Whether to track intermediate optimization data for visualization.
    - initial_samples (array-like): A set of initial samples to start the BO optimization with.
    - **kwargs: Algorithm-specific arguments for customization.
    
    Returns:
    - best_pose (np.ndarray): The optimized pose (x, y, a).
    - history (dict): A dictionary containing 'sampled_points', 'sampled_scores', and 'predicted_scores' (if track_info=True).
    """
    history = {
        'sampled_points': [], 'sampled_scores': [],
        'predicted_scores': [], 'rendered_images': [],
        'batch_size': batch_size,
        'n_iterations': n_iterations,
    } if track_info else None

    # Normalize bounds for [0, 1] range if requested
    if normalize_data:
        original_bounds = bounds
        bounds = [(0, 1)] * len(bounds)

        def normalized_F(poses):
            denormalized_poses = denormalize(poses, original_bounds)
            if len(denormalized_poses) < batch_size:
                return F(denormalized_poses)
            else:
                num_batches = len(denormalized_poses) // batch_size
                all_scores, all_info = [], {}
                for i in range(num_batches):
                    scores, info = F(denormalized_poses[i * batch_size:(i + 1) * batch_size])
                    all_scores.extend(scores)
                    if i == 0:
                        all_info = info
                    else:
                        for k in all_info.keys():
                            all_info[k] += info[k]
                return all_scores, all_info
    else:
        original_bounds = bounds
        normalized_F = F

    # Clip samples to bounds
    def clip_to_bounds(samples):
        return np.clip(samples, [b[0] for b in bounds], [b[1] for b in bounds])

    # Convert bounds into skopt format
    space = [Real(b[0], b[1], name=f'x{i}') for i, b in enumerate(bounds)]

    def objective_batch(X):
        X_clipped = clip_to_bounds(X)
        scores, score_info = normalized_F(np.array(X_clipped))
        if track_info:
            history['sampled_points'].extend(X_clipped)
            history['sampled_scores'].extend(scores)
            if "rendered_images" in score_info:
                history['rendered_images'].extend(score_info['rendered_images'])
        # we labelled -1, -2 scores for logging purposes
        # all scores sent to BO should be non-negative
        # so costs should be all negative
        cost = [-max(0, s) for s in scores]
        return cost

    if algorithm == 'bayesian':
        base_estimator = kwargs.get('base_estimator', 'GP')
        acq_func = kwargs.get('acq_func', 'EI')
        acq_func_kwargs = None
        use_dynamic_kappa = False
        if "acq_func_kwargs" in kwargs and "kappa" in kwargs["acq_func_kwargs"]:
            if type(kwargs["acq_func_kwargs"]["kappa"]) is float:
                acq_func_kwargs = dict(kappa=kwargs["acq_func_kwargs"]["kappa"])
            else:
                acq_func_kwargs = dict(kappa=kwargs["acq_func_kwargs"]["kappa"][0])
                kappa_start, kappa_end = kwargs["acq_func_kwargs"]["kappa"]
                alpha = np.log(kappa_start / kappa_end) / n_iterations
                def dynamic_kappa(t):
                    return kappa_start * np.exp(-alpha * t)
                use_dynamic_kappa = True
        opt = Optimizer(dimensions=space, base_estimator=base_estimator, acq_func=acq_func,
                        acq_func_kwargs=acq_func_kwargs,
                        n_jobs=-1, random_state=seed)

        # Add initial samples if provided
        if initial_samples is not None:
            initial_samples_clipped = clip_to_bounds(np.array(initial_samples))
            initial_costs = objective_batch(initial_samples_clipped)
            opt.tell(initial_samples_clipped.tolist(), initial_costs)

        pbar = tqdm(range(n_iterations))
        for t in pbar:
            batch_points = opt.ask(n_points=batch_size)
            batch_points_clipped = clip_to_bounds(batch_points)
            batch_costs = objective_batch(batch_points_clipped)
            opt.tell(batch_points_clipped.tolist(), batch_costs)

            if use_dynamic_kappa:
                opt.acq_func_kwargs["kappa"] = dynamic_kappa(t)
            
            # Store predicted scores for visualization
            if track_info and hasattr(opt.models[-1], "predict"):
                best_idx = np.argmax(history['sampled_scores'])
                best_pose = history['sampled_points'][best_idx]
                pbar.set_description(f"BO (Max Score: {np.max(history['sampled_scores']):.3f}, " + \
                                     f"Best Pose: {print_fn(denormalize(best_pose.round(2), original_bounds))}, " + \
                                     f"Kappa: {opt.acq_func_kwargs['kappa']:.2f})")

        best_pose = np.array(opt.Xi[np.argmin(opt.yi)])
        history['predicted_scores'] = None

    else:
        raise ValueError("Unknown algorithm: {}".format(algorithm))

    # Denormalize results if normalization was applied
    if normalize_data:
        best_pose = denormalize(best_pose, original_bounds)
        if track_info:
            history['sampled_points'] = denormalize(np.array(history['sampled_points']), original_bounds)

    return (best_pose, history) if track_info else best_pose
