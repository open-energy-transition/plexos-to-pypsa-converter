"""
Universal Progress Tracking for Network Setup Functions.

This module provides progress tracking and testing mode capabilities that can be
used across all network setup functions in the plexos-to-pypsa-converter package.
"""

import time
import sys
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class ProgressConfig:
    """Configuration for progress tracking and testing mode."""
    testing_mode: bool = False
    test_limit: int = 100
    show_progress: bool = True
    update_interval: int = 10  # Update progress every N items
    

class ProgressTracker:
    """
    Universal progress tracker for network setup operations.
    
    Provides progress bars, time tracking, and testing mode functionality
    that can be used across all network setup functions.
    
    Parameters
    ----------
    total_items : int
        Total number of items to process
    description : str
        Description of the operation (e.g., "Processing Generators")
    config : ProgressConfig, optional
        Configuration for progress tracking and testing mode
        
    Examples
    --------
    >>> config = ProgressConfig(testing_mode=True, test_limit=50)
    >>> tracker = ProgressTracker(1000, "Processing Facilities", config)
    >>> for i, item in enumerate(items):
    ...     if not tracker.should_process(i):
    ...         break
    ...     process_item(item)
    ...     tracker.update(i + 1, item.name)
    >>> tracker.finish()
    """
    
    def __init__(self, total_items: int, description: str, 
                 config: Optional[ProgressConfig] = None):
        self.total_items = total_items
        self.description = description
        self.config = config or ProgressConfig()
        
        # Determine actual number to process
        if self.config.testing_mode:
            self.items_to_process = min(total_items, self.config.test_limit)
            print(f"⚠️  TESTING MODE: {description} limited to {self.items_to_process}/{total_items}")
        else:
            self.items_to_process = total_items
            
        # Progress tracking state
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.current_item = 0
        self.last_progress_update = 0
        
        # Display initial progress
        if self.config.show_progress and self.items_to_process > 0:
            print(f"Starting {description}...")
            self._display_progress(0)
    
    def should_process(self, index: int) -> bool:
        """
        Check if item at given index should be processed.
        
        Parameters
        ----------
        index : int
            Zero-based index of the item
            
        Returns
        -------
        bool
            True if item should be processed, False if skipped due to testing mode
        """
        return index < self.items_to_process
    
    def update(self, current: int, item_name: Optional[str] = None) -> None:
        """
        Update progress tracker with current status.
        
        Parameters
        ----------
        current : int
            Current number of items processed (1-based)
        item_name : str, optional
            Name of current item being processed
        """
        self.current_item = current
        
        # Only update display periodically to avoid performance impact
        if (self.config.show_progress and 
            (current - self.last_progress_update >= self.config.update_interval or 
             current >= self.items_to_process)):
            
            self._display_progress(current, item_name)
            self.last_progress_update = current
    
    def _display_progress(self, current: int, item_name: Optional[str] = None) -> None:
        """Display current progress with progress bar and time estimates."""
        if self.items_to_process == 0:
            return
            
        # Calculate progress
        progress = current / self.items_to_process
        percentage = int(progress * 100)
        
        # Create progress bar
        bar_width = 20
        filled_width = int(progress * bar_width)
        bar = "█" * filled_width + "░" * (bar_width - filled_width)
        
        # Calculate time estimates
        elapsed_time = time.time() - self.start_time
        if current > 0:
            time_per_item = elapsed_time / current
            remaining_items = self.items_to_process - current
            eta_seconds = time_per_item * remaining_items
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."
        
        # Format current item info
        item_info = f" ({item_name})" if item_name else ""
        
        # Display progress line
        progress_line = (f"\r  {self.description}: {current}/{self.items_to_process} "
                        f"({percentage}%) [{bar}] ETA: {eta_str}{item_info}")
        
        # Truncate if too long for terminal
        if len(progress_line) > 100:
            progress_line = progress_line[:97] + "..."
            
        print(progress_line, end="", flush=True)
        
        # New line when complete
        if current >= self.items_to_process:
            print()
    
    def finish(self, items_added: Optional[int] = None) -> None:
        """
        Finish progress tracking and display summary.
        
        Parameters
        ----------
        items_added : int, optional
            Number of items successfully added (may be less than processed)
        """
        elapsed_time = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed_time)
        
        if items_added is not None:
            success_rate = (items_added / self.current_item * 100) if self.current_item > 0 else 0
            print(f"  ✓ {self.description} complete: {items_added} added "
                  f"({success_rate:.1f}% success) in {elapsed_str}")
        else:
            print(f"  ✓ {self.description} complete: {self.current_item} processed in {elapsed_str}")
            
        if self.config.testing_mode and self.total_items > self.items_to_process:
            skipped = self.total_items - self.items_to_process
            print(f"    Note: {skipped} items skipped in testing mode")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m{secs:02d}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes:02d}m"


def create_progress_tracker(total_items: int, description: str, 
                           testing_mode: bool = False, 
                           test_limit: int = 100,
                           show_progress: bool = True) -> ProgressTracker:
    """
    Convenience function to create a ProgressTracker.
    
    Parameters
    ----------
    total_items : int
        Total number of items to process
    description : str
        Description of the operation
    testing_mode : bool, default False
        Enable testing mode with limited processing
    test_limit : int, default 100
        Maximum items to process in testing mode
    show_progress : bool, default True
        Show progress bars and updates
        
    Returns
    -------
    ProgressTracker
        Configured progress tracker instance
    """
    config = ProgressConfig(
        testing_mode=testing_mode,
        test_limit=test_limit,
        show_progress=show_progress
    )
    return ProgressTracker(total_items, description, config)


class BatchProgressTracker:
    """
    Progress tracker for operations with multiple phases/batches.
    
    Useful for network setup functions that process multiple types of components
    sequentially (e.g., generators, then links, then storage).
    
    Examples
    --------
    >>> tracker = BatchProgressTracker("Setting up network", testing_mode=True)
    >>> 
    >>> gen_progress = tracker.add_batch(500, "Processing Generators")
    >>> for i, gen in enumerate(generators):
    ...     if not gen_progress.should_process(i):
    ...         break
    ...     process_generator(gen)
    ...     gen_progress.update(i + 1)
    >>> gen_progress.finish(generators_added)
    >>> 
    >>> link_progress = tracker.add_batch(200, "Processing Links")  
    >>> for i, link in enumerate(links):
    ...     if not link_progress.should_process(i):
    ...         break
    ...     process_link(link)
    ...     link_progress.update(i + 1)
    >>> link_progress.finish(links_added)
    >>> 
    >>> tracker.finish()
    """
    
    def __init__(self, operation_name: str, testing_mode: bool = False, 
                 show_progress: bool = True):
        self.operation_name = operation_name
        self.testing_mode = testing_mode
        self.show_progress = show_progress
        self.batches = []
        self.start_time = time.time()
        
        if testing_mode:
            print(f"⚠️  TESTING MODE: {operation_name} - processing limited subsets")
        
    def add_batch(self, total_items: int, description: str, 
                  test_limit: int = 100) -> ProgressTracker:
        """Add a new batch/phase to track."""
        config = ProgressConfig(
            testing_mode=self.testing_mode,
            test_limit=test_limit,
            show_progress=self.show_progress
        )
        tracker = ProgressTracker(total_items, description, config)
        self.batches.append(tracker)
        return tracker
    
    def finish(self) -> None:
        """Finish batch tracking and display overall summary."""
        total_time = time.time() - self.start_time
        time_str = ProgressTracker._format_time(total_time)
        
        print(f"\n✓ {self.operation_name} complete in {time_str}")
        
        if self.testing_mode:
            print("  ⚠️  Testing mode was enabled - model may be incomplete")