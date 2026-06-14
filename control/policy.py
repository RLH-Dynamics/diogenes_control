import onnxruntime as ort
import numpy as np
import time
import sys


class Policy:
    """Runs the MJLab-trained ONNX policy live on the robot.

    OBSERVATION LAYOUT  (must match the MJLab *actor* observation group exactly)
    --------------------------------------------------------------------------
    The deployed ONNX policy is the ACTOR. In env_cfgs.diogenes_env_cfg the actor
    group is built by _proprio_terms() and then has its slider terms DELETED:

        actor_terms = _proprio_terms(...)        # joint_pos, joint_vel,
                                                 # slider_pos, slider_vel,
                                                 # last_action, phase_clock
        del actor_terms["slider_pos"]            # privileged -> critic only
        del actor_terms["slider_vel"]            # privileged -> critic only

    This is an ASYMMETRIC actor-critic: the slider (carriage) state is given to
    the value function ONLY. The deployed actor never sees it, so the robot needs
    NO rail sensor.

    With concatenate_terms=True, mjlab's ObservationManager.compute_group cats the
    surviving terms in dict-insertion order (verified against the mjlab source:
    torch.cat(list(group_obs.values()), dim=-1)). No Diogenes term sets
    history_length, so there is NO history stacking. The resulting 11-dim vector
    is:

        [ joint_pos_rel  (3),    # joint_pos - default_pos, order (hip, thigh, calf)
          joint_vel_rel  (3),    # joint_vel - default_vel(=0), same order
          last_action    (3),    # previous RAW policy action (pre scale/offset)
          phase_clock    (2) ]   # [sin(2*pi*phi), cos(2*pi*phi)]

    Conventions this code must honor (all verified against mjlab source):
      * joint_pos_rel subtracts default_joint_pos, so `default_pos` here MUST equal
        the sim DEFAULT_INIT pose used in training (see the DEFAULT_POS note in
        config.py / the repo's diogenes_constants.py).
      * The JointPositionAction uses use_default_offset=True with scale=ACTION_SCALE,
        so the commanded ABSOLUTE joint target is  raw_action*scale + default_pos.
      * The observation's `last_action` is `action_manager.action`, documented in
        mjlab as the RAW policy output BEFORE per-term scale/offset. So we feed
        back the raw network output from the previous policy tick -- NOT the
        physical motor target. We store it in self.last_raw_action.
      * obs_normalization=True is baked INTO the exported ONNX graph (rsl-rl wraps
        the running mean/std normalizer as the first layer), so we feed RAW,
        UNNORMALIZED observations.
      * No obs scale/clip is configured on any Diogenes term, so none is applied.
    """

    # Expected flat observation width (see layout above).
    EXPECTED_OBS_DIM = 11

    def __init__(self, model_path, num_joints, period, default_pos,
                 direction_vector, action_scale):
        print(f"[INFO] Loading model: {model_path}")
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
        except Exception as e:
            print(f"[ERROR] Model load failed: {e}")
            sys.exit(1)

        self.num_joints = num_joints
        self.period = period
        self.default_pos = np.array(default_pos, dtype=np.float32)
        self.direction_vector = np.array(direction_vector, dtype=np.float32)
        self.action_scale = action_scale

        # Previous RAW policy action (pre scale/offset). At reset the action
        # history in mjlab is zeroed, so we start from zeros to match the very
        # first inference the policy saw in sim.
        self.last_raw_action = np.zeros(num_joints, dtype=np.float32)

        # Validate the ONNX input width against the layout we build. A mismatch
        # almost always means the policy was trained with a different obs set
        # (e.g. slider re-enabled on the actor, or a history buffer added) and
        # MUST be reconciled before running on hardware.
        self._verify_input_dim()

        self.start_time = time.perf_counter()

    def _verify_input_dim(self):
        """Hard-check the model's expected input width matches our layout."""
        try:
            shape = self.session.get_inputs()[0].shape  # e.g. [1, 11] or ['b', 11]
            width = shape[-1]
        except Exception as e:
            print(f"[WARN] Could not introspect model input shape: {e}")
            return
        if isinstance(width, int) and width != self.EXPECTED_OBS_DIM:
            print(
                f"[ERROR] Model expects obs width {width}, but this deploy code "
                f"builds {self.EXPECTED_OBS_DIM} "
                f"(joint_pos_rel 3 + joint_vel_rel 3 + last_action 3 + phase 2). "
                f"The observation layout does NOT match the trained policy. "
                f"Refusing to run with a mismatched observation vector."
            )
            sys.exit(1)
        print(f"[INFO] Observation width OK ({self.EXPECTED_OBS_DIM} dims).")

    def reset_phase(self):
        """Reset the phase clock origin and action history (call before a run)."""
        self.start_time = time.perf_counter()
        self.last_raw_action = np.zeros(self.num_joints, dtype=np.float32)

    def compute_action(self, state_vector):
        # 1. Extract and sort raw hardware states by motor ID. Sorting by key
        #    matches the (hip, thigh, calf) model-declaration order, since the
        #    JOINT_CONFIG ids are hip=1, thigh=2, knee/calf=3.
        sorted_keys = sorted(state_vector.keys())
        raw_pos = np.array([state_vector[k]['pos'] for k in sorted_keys],
                           dtype=np.float32)
        raw_vel = np.array([state_vector[k]['vel'] for k in sorted_keys],
                           dtype=np.float32)

        # 2. Input transform (real -> sim): apply the per-joint direction flip,
        #    then make position RELATIVE to the default pose (joint_pos_rel).
        #    joint_vel_rel subtracts default_vel which is 0, so velocity is just
        #    the sim-frame velocity.
        sim_pos = raw_pos * self.direction_vector
        sim_vel = raw_vel * self.direction_vector
        rel_pos = sim_pos - self.default_pos          # joint_pos_rel   (3,)
        rel_vel = sim_vel                             # joint_vel_rel   (3,)

        # 3. Phase clock: sin/cos of the global hop phase, advancing with
        #    wall-clock time and wrapping every `period` seconds. Matches
        #    diogenes_mdp.phase_clock (angle = 2*pi*phi, phi = t/period mod 1).
        elapsed = time.perf_counter() - self.start_time
        angle = 2.0 * np.pi * (elapsed / self.period)
        phase_signal = [np.sin(angle), np.cos(angle)]   # (2,)

        # 4. Assemble the observation in the EXACT trained order. No history.
        obs = np.concatenate([
            rel_pos,                                 # 0:3  joint_pos_rel
            rel_vel,                                 # 3:6  joint_vel_rel
            self.last_raw_action,                    # 6:9  last_action (raw)
            np.asarray(phase_signal, dtype=np.float32),  # 9:11 phase_clock
        ]).astype(np.float32).reshape(1, -1)

        # 5. Inference (normalization is inside the ONNX graph; feed raw obs).
        raw_actions = self.session.run(None, {self.input_name: obs})[0][0]
        raw_actions = np.asarray(raw_actions, dtype=np.float32)

        # 6. Store this step's RAW action for next step's last_action observation
        #    (mjlab feeds back the pre-scale/offset network output).
        self.last_raw_action = raw_actions.copy()

        # 7. Output transform (sim -> real): apply scale + default offset
        #    (use_default_offset=True), then the per-joint direction flip back to
        #    hardware frame.
        target_pos_sim = raw_actions * self.action_scale + self.default_pos
        physical_targets = target_pos_sim * self.direction_vector

        return physical_targets
