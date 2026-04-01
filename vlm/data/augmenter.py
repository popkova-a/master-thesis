import random
from typing import List, Tuple, Callable, Optional

import torch


class EventAugmenter:
    def __init__(self,
                 resolution: Tuple[int, int] = (128, 128),  # Height, width
                 max_shift: int = 50,
                 hor_flip_p: float = 0.5,
                 time_flip_p: float = 0.8,
                 center_p: float = 0.5,
                 time_crop_p: float = 0.8,
                 time_drop_ratio: float = 0.1,
                 area_drop_ratio: float = 0.1,
                 point_drop_ratio: float = 0.1,
                 noise_std: float = 0.1,
                 include_augmentations: Optional[List[str]] = None,
                 min_event_num: Optional[int] = 1000,
                 verbose: bool = True):
        """
        Initialize the event data augmentation pipeline.

        Args:
            resolution: Sensor resolution as (height, width) in pixels. Determines the valid
                spatial range for event coordinates after augmentation.

            max_shift: Maximum absolute value for random spatial shifts (pixels). Events will
                be shifted by a random value in [-max_shift, max_shift] for both x and y axes.
                Must be between 0 and max(resolution). Default: 50.

            hor_flip_p: Probability [0, 1] of applying horizontal flip. When triggered, events
                are mirrored across the vertical center axis. Default: 0.5.

            time_flip_p: Probability [0, 1] of reversing the temporal order of events. When
                triggered, the event stream is reversed and polarities are inverted to maintain
                brightness change semantics. Default: 0.8.

            center_p: Probability [0, 1] of centering events. When triggered, events are
                spatially centered around the image center and temporally aligned to start at t=0.
                Default: 0.5.

            time_crop_p: Probability [0, 1] of applying random temporal cropping. When triggered,
                a continuous time window containing a random fraction of events is selected.
                Default: 0.8.

            time_drop_ratio: Fraction [0, 1) of events to drop from a random continuous time
                window. The window length is time_drop_ratio * total duration. Default: 0.1.

            area_drop_ratio: Fraction [0, 1) of sensor area to drop events from. A rectangular
                region covering this fraction of total area is randomly selected and events
                within it are removed. Default: 0.1.

            point_drop_ratio: Probability [0, 1) of dropping individual events. Each event has
                this independent probability of being removed. Default: 0.1.

            noise_std: Standard deviation of Gaussian noise added to event coordinates (t, x, y).
                Noise is clipped to maintain valid spatial coordinates. Set to 0 to disable.
                Default: 0.1.

            include_augmentations: List of augmentation names to include in the pipeline.
                If None (default), all available augmentations are used. Available options:
                ['shift', 'flip', 'time_flip', 'center', 'time_crop', 'time_drop',
                 'area_drop', 'point_drop', 'noise']

            min_event_num (int): Minimum number of events in the resulting event stream.

            verbose: Whether to print details about applied augmentations. Useful for debugging.
                Default: True.

        Raises:
            ValueError: If any parameter values are invalid (e.g., probabilities outside [0,1],
                negative noise std, or invalid max_shift).
        """

        self.resolution = resolution
        self.max_shift = max_shift
        self.hor_flip_p = hor_flip_p
        self.time_flip_p = time_flip_p
        self.center_p = center_p
        self.time_crop_p = time_crop_p
        self.time_drop_ratio = time_drop_ratio
        self.area_drop_ratio = area_drop_ratio
        self.point_drop_ratio = point_drop_ratio
        self.noise_std = noise_std
        self.min_event_num = min_event_num
        self.verbose = verbose

        # Define available augmentations and their methods
        self.augmentation_map = {'shift': self.random_shift,
                                 'flip': self.random_horizontal_flip,
                                 'time_flip': self.random_time_flip,
                                 'center': self.random_center,
                                 'time_crop': self.random_time_crop,
                                 'time_drop': self.random_time_drop,
                                 'area_drop': self.random_area_drop,
                                 'point_drop': self.random_point_drop,
                                 'noise': self.add_noise}

        # Set which augmentations to include
        self.include_augmentations = (list(self.augmentation_map.keys())
                                      if include_augmentations is None
                                      else include_augmentations)

        # Validate the shift value
        if self.max_shift < 0 or self.max_shift > max(resolution):
            raise ValueError(f"max_shift should lie between 0 and {max(resolution)}")

        # Validate the probability value
        valid_prob = [0 <= hor_flip_p <= 1, 0 <= time_flip_p <= 1, 0 <= center_p <= 1, 0 <= time_crop_p <= 1]
        if not all(valid_prob):
            raise ValueError("Probability must be in range [0, 1].")

        # Validate the drop ratio
        valid_drop = [0 <= time_drop_ratio < 1, 0 <= area_drop_ratio < 1, 0 <= point_drop_ratio < 1]
        if not all(valid_drop):
            raise ValueError("Drop ratio must be in range [0, 1).")

        # Validate the noise std
        valid_std = noise_std > 0
        if not valid_std:
            raise ValueError("Noise standard deviation must be non-negative.")

    def _shift(self,
               events: torch.Tensor,
               x_shift: int,
               y_shift: int) -> torch.Tensor:
        """
        Shifts event coordinates while maintaining spatial validity.

        Only spatial coordinates (x,y) are modified; timestamps and polarities remain unchanged.
        Events that fall outside the sensor resolution after shifting are removed.

        Args:
            events (torch.Tensor): Event tensor in format (t, x, y, p), shape (N, 4), where:
                    t (float): Timestamp of the event
                    x (int): x-coordinate (column index, 0 <= x < width)
                    y (int): y-coordinate (row index, 0 <= y < height)
                    p (int): Polarity (0 or 1)
            x_shift (int): Horizontal shift of the event stream
            y_shift (int): Vertical shift of the event stream

        Returns:
            torch.Tensor: Shifted events in same (t, x, y, p) format, with out-of-bounds events removed.

        Raises:
            ValueError: If events tensor does not have shape (N, 4).

        Note:
            The function returns the original event stream if the shift pushes all the events
            out of the bounds.
        """

        if events.shape[1] != 4:
            raise ValueError("Events must contain (t, x, y, p).")

        height, width = self.resolution

        # Apply shifts
        shifted_events = events.clone()
        shifted_events[:, 1] += x_shift
        shifted_events[:, 2] += y_shift

        # Filter valid events
        valid_x = (shifted_events[:, 1] >= 0) & (shifted_events[:, 1] < width)
        valid_y = (shifted_events[:, 2] >= 0) & (shifted_events[:, 2] < height)
        valid_mask = valid_x & valid_y

        # Return the original event stream if the shift leaves no valid events
        if not valid_mask.any() or valid_mask.sum() < self.min_event_num:
            return events
        else:
            return shifted_events[valid_mask].contiguous()

    def random_shift(self,
                     events: torch.Tensor) -> torch.Tensor:
        """
        Applies a random spatial shift to event coordinates while maintaining validity within sensor
        resolution. Randomly shifts event coordinates along both x and y axes within the range
        [-max_shift, max_shift].

        Args:
            events (torch.Tensor): Event tensor in format (t, x, y, p), shape (N, 4)

        Returns:
            torch.Tensor: Shifted events in same (t, x, y, p) format, with out-of-bounds events removed.

        Raises:
            ValueError: If events tensor does not have shape (N, 4).

        Note:
            - The function returns the original event stream if the shift pushes all the events
              out of the bounds.
            - Only spatial coordinates (x,y) are modified; timestamps and polarities remain unchanged.
              Events that fall outside the sensor resolution after shifting are removed.
        """

        # Generate independent random shifts for x and y axes within configured bounds
        x_shift = random.randint(-self.max_shift, self.max_shift)
        y_shift = random.randint(-self.max_shift, self.max_shift)

        if self.verbose:
            print(f"\t -> Random shift by {x_shift} in x and {y_shift} in y")

        # Delegate the actual shifting operation to the deterministic _shift method
        return self._shift(events=events,
                           x_shift=x_shift,
                           y_shift=y_shift)

    def _horizontal_flip(self,
                         events: torch.Tensor) -> torch.Tensor:
        """
        Flips event coordinates horizontally.

        Designed for event-based vision data in the format (t, x, y, p).
        Only the x-coordinates are modified; y-coordinates, timestamps, and polarities remain unchanged.
        The flip is performed by mirroring events across the vertical center axis of the sensor.

        Args:
            events (torch.Tensor): Event tensor in format (t, x, y, p), shape (N, 4)

        Returns:
            torch.Tensor: Events in same (t, x, y, p) format, with x-coordinates flipped.

        Raises:
            ValueError: If events tensor does not have shape (N, 4).
        """

        if events.shape[1] != 4:
            raise ValueError("Events must contain (t, x, y, p).")

        height, width = self.resolution

        # Mirror x-coordinates
        flipped_events = events.clone()
        flipped_events[:, 1] = width - 1 - events[:, 1]

        return flipped_events.contiguous()

    def random_horizontal_flip(self,
                               events: torch.Tensor) -> torch.Tensor:
        """
        Randomly applies horizontal flipping to event coordinates based on hor_flip_p
        probability.

        Args:
            events (torch.Tensor): Event tensor in format (t, x, y, p) with shape (N, 4)

        Returns:
            torch.Tensor: Horizontally flipped events if triggered, otherwise original events
        """

        if self.verbose:
            print("\t -> Random horizontal flip")

        # Generate random number to decide whether to flip
        if random.random() < self.hor_flip_p:
            # Apply horizontal flip if probability condition is met
            return self._horizontal_flip(events=events)
        else:
            # Return original events otherwise
            return events

    @staticmethod
    def _time_flip(events: torch.Tensor) -> torch.Tensor:
        """
        Reverses the temporal order of an event stream.

        For event cameras generating data in (t, x, y, p) format, this operation:
        1. Reverses the event sequence order (last event becomes first)
        2. Adjusts timestamps to maintain relative temporal differences
        3. Inverts polarities to preserve correct brightness change semantics

        Args:
            events (torch.Tensor): Event tensor in (t, x, y, p) format, shape (N,4)

        Returns:
            torch.Tensor: Time-flipped events.

        Raises:
            ValueError: If events tensor does not have shape (N, 4).
        """

        if events.shape[1] != 4:
            raise ValueError("Events must contain (t, x, y, p).")

        # Reverse the event order
        flipped_events = torch.flip(events.clone(),
                                    dims=[0])

        # Preserve relative intervals
        t_start = flipped_events[0, 0]
        flipped_events[:, 0] = t_start - flipped_events[:, 0]

        # Polarity inversion (0 -> 1, 1 -> 0)
        flipped_events[:, 3] = 1 - flipped_events[:, 3]

        return flipped_events.contiguous()

    def random_time_flip(self,
                         events: torch.Tensor) -> torch.Tensor:
        """
        Randomly applies temporal reversal to event stream based on time_flip_p probability.

        Args:
           events (torch.Tensor): Event tensor in (t, x, y, p) format, shape (N,4)

        Returns:
           torch.Tensor: Time-flipped events if triggered, otherwise original events
        """

        if self.verbose:
            print("\t -> Random temporal flip")

        # Check probability threshold for time flip
        if random.random() < self.time_flip_p:
            # Apply temporal reversal and polarity inversion
            return self._time_flip(events=events)
        else:
            # Return original events otherwise
            return events

    def _center(self,
                events: torch.Tensor) -> torch.Tensor:
        """
        Centers event coordinates both spatially and temporally.

        Performs two normalization operations:
        1. Temporal: Shifts timestamps so the first event occurs at t=0
        2. Spatial: Centers the event cloud around the image center

        Args:
            events (torch.Tensor): Event tensor in format (t, x, y, p), shape (N, 4)

        Returns:
            torch.Tensor: Centered event tensor with same shape as input.

        Raises:
            ValueError: If events tensor does not have shape (N, 4)

        Notes:
            - Spatial centering: Ensures (min_x + max_x)/2 = W/2 and (min_y + max_y)/2 = H/2
            - Temporal centering: Makes min_t = 0 while preserving relative timestamps
        """

        if events.shape[1] != 4:
            raise ValueError("Events must contain (t, x, y, p).")

        height, width = self.resolution

        # Temporal centering
        centered_events = events.clone()
        centered_events[:, 0] -= centered_events[:, 0].min()

        # Spatial centering
        x_min, x_max = centered_events[:, 1].min(), centered_events[:, 1].max()
        y_min, y_max = centered_events[:, 2].min(), centered_events[:, 2].max()
        x_shift = ((x_max + x_min + 1.) - width) // 2.
        y_shift = ((y_max + y_min + 1.) - height) // 2.
        centered_events[:, 1] -= x_shift
        centered_events[:, 2] -= y_shift

        return centered_events.contiguous()

    def random_center(self,
                      events: torch.Tensor) -> torch.Tensor:
        """
        Randomly centers event coordinates both spatially and temporally based on center_p probability.

        Args:
            events (torch.Tensor): Event tensor in (t, x, y, p) format, shape (N,4)

        Returns:
            torch.Tensor: Centered events if triggered, otherwise original events

        Note:
            Spatial centering ensures:
            - (min_x + max_x)/2 = width/2
            - (min_y + max_y)/2 = height/2
            Temporal centering makes min_t = 0 while preserving relative timestamps
        """

        if self.verbose:
            print("\t -> Random centering (temporal and spatial)")

        # Check probability threshold for centering
        if random.random() < self.center_p:
            # Apply temporal reversal and polarity inversion
            return self._center(events=events)
        else:
            # Return original events otherwise
            return events

    def _time_crop(self,
                   events: torch.Tensor,
                   start_ratio: float,
                   end_ratio: float) -> torch.Tensor:
        """
        Crops events within a specific time window defined by start and end ratios.

        Args:
            events (torch.Tensor): Event tensor in (t, x, y, p) format, shape (N,4)
            start_ratio (float): Start time of the crop window (ratio of total duration)
            end_ratio (float): End time of the crop window (ratio of total duration)

        Returns:
            torch.Tensor: Cropped event tensor containing only events within the time window.

        Raises:
            ValueError: If events tensor does not have shape (N, 4)
                       or if time ratios are invalid.

         Note:
            The function returns the original event stream if the crop removes all the events.
        """

        if events.shape[1] != 4:
            raise ValueError("Events must contain (t, x, y, p).")

        if not (0 <= start_ratio < end_ratio <= 1):
            raise ValueError("Invalid time window ratios. Must satisfy 0 <= start_ratio < end_ratio <= 1")

        # Get timestamps and total duration
        timestamps = events[:, 0]
        max_t = torch.max(timestamps)

        # Calculate absolute time bounds
        t_start = max_t * start_ratio
        t_end = max_t * end_ratio

        # Keep events within the time window
        mask = (timestamps >= t_start) & (timestamps <= t_end)

        # Return the original event stream if the crop leaves no valid events
        if not mask.any() or mask.sum() < self.min_event_num:
            return events
        else:
            return events[mask].contiguous()

    def random_time_crop(self,
                         events: torch.Tensor) -> torch.Tensor:
        """
        Randomly crops a continuous time window from the event stream.
        The relative length of the window is sampled uniformly.

        Args:
            events (torch.Tensor): Event tensor in (t, x, y, p) format, shape (N,4)

        Returns:
            torch.Tensor: Cropped event tensor containing only events within the random time window.
        """

        if self.verbose:
            print(f"\t -> Random time crop")

        # Check probability threshold for time crop
        if random.random() < self.time_crop_p:
            # Sample crop ratio uniformly
            crop_ratio = random.uniform(0, 1)

            # Calculate maximum possible start time
            max_start = 1 - crop_ratio

            # Randomly select start time
            start_ratio = random.uniform(0, max_start)

            # Define corresponding end time
            end_ratio = start_ratio + crop_ratio

            return self._time_crop(events=events,
                                   start_ratio=start_ratio,
                                   end_ratio=end_ratio)
        else:
            # Return original events otherwise
            return events

    def random_time_drop(self,
                         events: torch.Tensor) -> torch.Tensor:
        """
        Randomly drops events within a continuous time interval from the event stream corresponding
        to time_drop_ratio. Selects a random time window of relative length time_drop_ratio and removes
        all events that fall within it.

        Args:
            events (torch.Tensor): Event tensor in (t, x, y, p) format, shape (N,4)

        Returns:
            torch.Tensor: Filtered event array with events outside the selected time window,
                          or original events if time_drop_ratio is 0.

        Raises:
            ValueError: If events tensor does not have shape (N, 4)

         Note:
            The function returns the original event stream if the drop leaves no events.
        """

        if self.verbose:
            print("\t -> Random temporal drop")

        if events.shape[1] != 4:
            raise ValueError("Events must contain (t, x, y, p).")

        # Return original events if time_drop_ratio == 0
        if self.time_drop_ratio == 0:
            return events

        # Get timestamps
        timestamps = events[:, 0]
        max_t = torch.max(timestamps)

        # Random window selection (ensure it stays within [0, 1])
        t_start = random.uniform(0, 1 - self.time_drop_ratio)
        t_end = t_start + self.time_drop_ratio

        # Keep events outside the selected time window
        mask = (timestamps < max_t * t_start) | (timestamps > max_t * t_end)

        # Return the original event stream if the drop leaves no valid events
        if not mask.any() or mask.sum() < self.min_event_num:
            return events
        else:
            return events[mask].contiguous()

    def random_area_drop(self,
                         events: torch.Tensor) -> torch.Tensor:
        """
        Randomly drops events within a rectangular region of the sensor area.

        The dropped region covers a fraction of the total sensor area determined by
        self.area_drop_ratio. The region is centered at a random position and scaled
        such that:
            - area_drop_ratio = 0.25 → drops 25% of total area (0.5 × 0.5 linear dimensions)
            - area_drop_ratio = 0.50 → drops 50% of total area (~0.707 × 0.707 linear dimensions)

        Args:
            events (torch.Tensor): Event tensor in (t, x, y, p) format, shape (N,4)

        Returns:
            torch.Tensor: Filtered event array with events outside the dropped region,
                          or original events if area_drop_ratio is 0.

        Raises:
            ValueError: If events tensor does not have shape (N, 4)

        Note:
            The function returns the original event stream if the drop leaves no events.
        """

        if self.verbose:
            print("\t -> Random spatial drop")

        if events.shape[1] != 4:
            raise ValueError("Events must contain (t, x, y, p).")

        # Return original events if area_drop_ratio == 0
        if self.area_drop_ratio == 0:
            return events

        height, width = self.resolution

        # Calculate required linear dimension to achieve target area ratio
        linear_drop_ratio = self.area_drop_ratio ** 0.5

        # Random center point
        center_x = random.uniform(0, width)
        center_y = random.uniform(0, height)

        # Calculate region dimensions
        region_w = width * linear_drop_ratio
        region_h = height * linear_drop_ratio

        # Calculate bounded coordinates
        x0 = max(0, int(center_x - region_w / 2))
        y0 = max(0, int(center_y - region_h / 2))
        x1 = min(width, int(center_x + region_w / 2))
        y1 = min(height, int(center_y + region_h / 2))

        # Keep events outside the selected region
        mask_x = (events[:, 1] < x0) | (events[:, 1] > x1)
        mask_y = (events[:, 2] < y0) | (events[:, 2] > y1)
        mask = mask_x | mask_y

        # Return the original event stream if the drop leaves no valid events
        if not mask.any() or mask.sum() < self.min_event_num:
            return events
        else:
            return events[mask].contiguous()

    def random_point_drop(self,
                          events: torch.Tensor) -> torch.Tensor:
        """
        Randomly drops individual events from the event stream with point_drop_ratio probability.

        Args:
            events (torch.Tensor): Event tensor in (t, x, y, p) format, shape (N,4)

        Returns:
            torch.Tensor: Event tensor with random events removed
                          or original events if point_drop_ratio is 0.

        Note:
            The function returns the original event stream if the drop leaves no events.
        """

        if self.verbose:
            print("\t -> Random event drop")

        if events.shape[1] != 4:
            raise ValueError("Events must contain (t, x, y, p).")

        # Return original events if area_drop_ratio == 0
        if self.point_drop_ratio == 0:
            return events

        # Sample events to keep with 1 - self.point_drop_ratio probability
        mask = torch.rand(len(events)) > self.point_drop_ratio

        # Return the original event stream if the drop leaves no valid events
        if not mask.any() or mask.sum() < self.min_event_num:
            return events
        else:
            return events[mask].contiguous()

    def add_noise(self,
                  events: torch.Tensor,
                  eps: float = 1.0e-6) -> torch.Tensor:
        """
        Adds Gaussian noise to event coordinates and timestamps.

        Args:
            events (torch.Tensor): Event tensor in (t, x, y, p) format, shape (N,4)
            eps (float): Epsilon value.

        Returns:
            torch.Tensor: Event tensor with added noise.

        Raises:
            ValueError: If events tensor does not have shape (N, 4)
        """

        if self.verbose:
            print("\t -> Add Gaussian noise")

        if events.shape[1] != 4:
            raise ValueError("Events must contain (t, x, y, p).")

        if self.noise_std == 0:
            return events

        # Generate noise for all coordinates except polarity
        noisy_events = events.clone().float()
        noise = torch.randn_like(noisy_events[:, :3]) * self.noise_std

        # Apply spatial noise to x, y coordinates
        noisy_events[:, 1:3] += noise[:, 1:3]

        # Process temporal noise with normalized timestamps
        t = noisy_events[:, 0]
        t_min, t_max = t.min(), t.max()
        duration = t_max - t_min

        # Normalize time to [0, 1] range and add noise
        t_norm = (t - t_min) / (duration + eps)
        t_noisy = t_norm + noise[:, 0]

        # Renormalize noisy timestamps and scale back to original range
        t_noisy = (t_noisy - t_noisy.min()) / (t_noisy.max() - t_noisy.min() + eps)
        t_noisy = torch.clamp(t_noisy, 0, 1 - eps) * duration + t_min

        # Sort events by new timestamps and clamp to original time bounds
        sorted_idx = torch.argsort(t_noisy)
        noisy_events = noisy_events[sorted_idx]
        noisy_events[:, 0] = torch.clamp(t_noisy[sorted_idx], t_min, t_max)

        # Round spatial coordinates to nearest integer and clamp to valid range
        height, width = self.resolution
        noisy_events[:, 0] = torch.clamp(noisy_events[:, 0], 0)
        noisy_events[:, 1] = torch.clamp(noisy_events[:, 1], 0, width - 1)
        noisy_events[:, 2] = torch.clamp(noisy_events[:, 2], 0, height - 1)

        return noisy_events

    def get_pipeline(self) -> List[Callable]:
        """
        Constructs a family of augmentations that will be sampled and applied sequentially
        to event data in __call__ method.

        Returns:
            list: list of augmentation functions to apply, in random order.
        """

        # Only include selected augmentations
        augmentations = [self.augmentation_map[name]
                         for name in self.include_augmentations
                         if name in self.augmentation_map
                         and 'noise' not in name]
        random.shuffle(augmentations)

        if 'noise' in self.include_augmentations:
            augmentations.append(self.add_noise)

        return augmentations

    def __call__(self,
                 events: torch.Tensor) -> torch.Tensor:
        """
        Applies the selected augmentations a in random order.

        Args:
            events (torch.Tensor): Event tensor in (t, x, y, p) format, shape (N,4)

        Returns:
            torch.Tensor: Event tensor altered by the selected augmentations.
        """

        output = events.clone()

        if self.verbose:
            print("\nApplying the following augmentations:")

        # Sequentially apply the transforms
        for transform in self.get_pipeline():
            output = transform(output)

        if self.verbose:
            print('\n', "=" * 50)

        return output


def build_augmenter(config: dict) -> EventAugmenter:
    """
    Constructs an EventAugmenter instance from the configuration of the experiment.

    Args:
        config (dict): Dictionary containing the configuration of the experiment

    Returns:
        EventAugmenter: Configured event augmentation instance.

    Raises:
        KeyError: If required configuration keys are missing.
        ValueError: If any parameter values are invalid (handled by EventAugmenter).
    """

    aug_config = config['data']['augmentation']
    return EventAugmenter(resolution=config['data']['sensor_size'],
                          max_shift=aug_config['max_shift'],
                          hor_flip_p=aug_config['hor_flip_p'],
                          time_flip_p=aug_config['time_flip_p'],
                          center_p=aug_config['center_p'],
                          time_crop_p=aug_config['time_crop_p'],
                          time_drop_ratio=aug_config['time_drop_ratio'],
                          area_drop_ratio=aug_config['area_drop_ratio'],
                          point_drop_ratio=aug_config['point_drop_ratio'],
                          noise_std=aug_config['noise_std'],
                          include_augmentations=aug_config.get('include_augmentations'),
                          min_event_num=aug_config.get('min_event_num', 1000),
                          verbose=aug_config['verbose'])
