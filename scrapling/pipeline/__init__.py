"""Pipeline Framework - Core abstractions for fetch-parse-save cycles.

This module provides reusable Pipeline wrapper classes for implementing
web scraping workflows with stage-based processing.

Example:
    >>> pipeline = Pipeline()
    >>> result = pipeline.fetch(url).parse(selector).save(format='json')
    >>> print(result.data)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from typing import Optional

T = TypeVar("T")
R = TypeVar("R")


class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineError:
    """Error that occurred during pipeline execution.

    Attributes:
        stage: Name of the stage where the error occurred
        message: Human-readable error message
        exception: The original exception (if any)
        timestamp: When the error occurred
    """
    stage: str
    message: str
    exception: Optional[Exception] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return f"[{self.stage}] {self.message}"


@dataclass
class PipelineMetadata:
    """Metadata about pipeline execution.

    Attributes:
        url: The URL that was processed
        start_time: When the pipeline started
        end_time: When the pipeline ended (None if still running)
        stages_completed: List of completed stage names
        current_stage: Current stage being executed
    """
    url: str | None = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    stages_completed: list[str] = field(default_factory=list)
    current_stage: str | None = None

    @property
    def duration(self) -> float | None:
        """Duration of execution in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def is_complete(self) -> bool:
        """Whether the pipeline has completed."""
        return self.end_time is not None


@dataclass
class PipelineResult(Generic[T]):
    """Result of pipeline execution.

    Attributes:
        data: The main data extracted/processed by the pipeline
        metadata: Execution metadata including timing
        errors: List of errors encountered during execution
        stage_results: Results from each stage
    """
    data: T | None = None
    metadata: PipelineMetadata = field(default_factory=PipelineMetadata)
    errors: list[PipelineError] = field(default_factory=list)
    stage_results: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether the pipeline completed successfully."""
        return len(self.errors) == 0 and self.metadata.is_complete

    @property
    def timing(self) -> dict[str, float]:
        """Timing information for each stage."""
        timing_info = {}
        for stage_name, result in self.stage_results.items():
            if isinstance(result, dict) and "duration" in result:
                timing_info[stage_name] = result["duration"]
        return timing_info

    def add_error(self, stage: str, message: str, exception: Exception | None = None) -> None:
        """Add an error to the result."""
        error = PipelineError(stage=stage, message=message, exception=exception)
        self.errors.append(error)


# Stage hook types
StageHook = Callable[["PipelineContext", Any], Any]
StageHookWithoutArg = Callable[["PipelineContext"], Any]


class PipelineContext:
    """Context passed between pipeline stages.

    This object carries data and state through the pipeline stages,
    allowing stages to communicate and share state.

    Attributes:
        data: The main data being processed
        raw_response: The raw response from fetch stage
        parsed_data: Data after parse stage
        metadata: Pipeline metadata
        session: Optional session object for persistence
    """

    def __init__(self) -> None:
        self.data: Any = None
        self.raw_response: Any = None
        self.parsed_data: Any = None
        self.metadata: PipelineMetadata = PipelineMetadata()
        self.session: Any = None
        self._stage_outputs: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Set a value in the context."""
        self._stage_outputs[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context."""
        return self._stage_outputs.get(key, default)

    def clear(self) -> None:
        """Clear all context data."""
        self.data = None
        self.raw_response = None
        self.parsed_data = None
        self._stage_outputs.clear()


class PipelineStage(ABC):
    """Base class for pipeline stages.

    Each stage represents a single step in the pipeline (fetch, parse, save).
    Stages can have before/after hooks for customization.

    Attributes:
        name: Name of the stage
        enabled: Whether the stage is enabled
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        self._before_hooks: list[StageHookWithoutArg] = []
        self._after_hooks: list[StageHook] = []

    def before(self, hook: StageHookWithoutArg) -> "PipelineStage":
        """Add a hook to run before this stage.

        Args:
            hook: Function to run before the stage executes

        Returns:
            Self for chaining
        """
        self._before_hooks.append(hook)
        return self

    def after(self, hook: StageHook) -> "PipelineStage":
        """Add a hook to run after this stage.

        Args:
            hook: Function to run after the stage executes

        Returns:
            Self for chaining
        """
        self._after_hooks.append(hook)
        return self

    def _run_before_hooks(self, context: PipelineContext) -> None:
        """Run all before hooks."""
        for hook in self._before_hooks:
            hook(context)

    def _run_after_hooks(self, context: PipelineContext, result: Any) -> Any:
        """Run all after hooks."""
        for hook in self._after_hooks:
            result = hook(context, result)
        return result

    @abstractmethod
    def execute(self, context: PipelineContext) -> Any:
        """Execute the stage.

        Args:
            context: Pipeline context

        Returns:
            Stage-specific result
        """
        pass

    def run(self, context: PipelineContext) -> Any:
        """Run the stage with hooks.

        Args:
            context: Pipeline context

        Returns:
            Stage result after hooks
        """
        if not self.enabled:
            return None

        context.metadata.current_stage = self.name

        # Run before hooks
        self._run_before_hooks(context)

        # Execute stage
        start_time = time.perf_counter()
        try:
            result = self.execute(context)
            duration = time.perf_counter() - start_time

            # Run after hooks
            result = self._run_after_hooks(context, result)

            # Store in stage results
            context.metadata.stages_completed.append(self.name)
            context._stage_outputs[self.name] = {
                "result": result,
                "duration": duration,
                "status": StageStatus.COMPLETED.value
            }

            return result

        except Exception as e:
            duration = time.perf_counter() - start_time
            context._stage_outputs[self.name] = {
                "error": str(e),
                "duration": duration,
                "status": StageStatus.FAILED.value
            }
            raise PipelineStageError(
                stage=self.name,
                message=str(e),
                original=e
            )


class PipelineStageError(Exception):
    """Error that occurred during pipeline stage execution.

    This exception carries context about which stage failed,
    making it easier to debug pipeline errors.

    Attributes:
        stage: Name of the stage that failed
        message: Error message
        original: Original exception
    """

    def __init__(
        self,
        stage: str,
        message: str,
        original: Exception | None = None
    ) -> None:
        self.stage = stage
        self.message = message
        self.original = original
        super().__init__(f"Stage '{stage}' failed: {message}")


class FetchStage(PipelineStage):
    """Stage for fetching web content.

    This stage handles HTTP requests and retrieves web content.
    Can be customized with different fetchers.

    Attributes:
        url: URL to fetch
        fetcher: Optional custom fetcher instance
        kwargs: Additional arguments for the fetcher
    """

    def __init__(
        self,
        url: str | None = None,
        fetcher: Any = None,
        **kwargs: Any
    ) -> None:
        super().__init__(name="fetch")
        self.url = url
        self.fetcher = fetcher
        self.fetcher_kwargs = kwargs

    def execute(self, context: PipelineContext) -> Any:
        """Execute the fetch stage.

        Args:
            context: Pipeline context

        Returns:
            Response content
        """
        from scrapling import Fetcher

        url = self.url or context.metadata.url
        if not url:
            raise PipelineStageError(
                stage=self.name,
                message="No URL provided for fetch"
            )

        context.metadata.url = url

        # Use custom fetcher or default
        fetcher = self.fetcher or Fetcher()

        # Perform fetch
        response = fetcher.get(url, **self.fetcher_kwargs)

        # Store raw response in context
        context.raw_response = response

        return response


class ParseStage(PipelineStage):
    """Stage for parsing fetched content.

    This stage extracts data from fetched HTML/content using CSS selectors.

    Attributes:
        selector: CSS selector to extract data
        extract_all: Whether to extract all matches
        parser: Optional custom parser instance
    """

    def __init__(
        self,
        selector: str | None = None,
        extract_all: bool = False,
        parser: Any = None
    ) -> None:
        super().__init__(name="parse")
        self.selector = selector
        self.extract_all = extract_all
        self.parser = parser

    def execute(self, context: PipelineContext) -> Any:
        """Execute the parse stage.

        Args:
            context: Pipeline context

        Returns:
            Parsed data
        """
        if context.raw_response is None:
            raise PipelineStageError(
                stage=self.name,
                message="No content to parse. Run fetch stage first."
            )

        response = context.raw_response

        # If no selector, return the whole response
        if not self.selector:
            context.parsed_data = response
            return response

        # Use the response's parser
        if hasattr(response, "ast"):
            parsed = response.ast(self.selector)
            if self.extract_all:
                result = parsed.all() if hasattr(parsed, 'all') else [parsed]
            else:
                result = parsed.first() if hasattr(parsed, 'first') else parsed
        else:
            # Fallback: try using the response directly
            result = response

        context.parsed_data = result
        return result


class SaveStage(PipelineStage):
    """Stage for saving parsed data.

    This stage saves the extracted data to a file or other storage.

    Attributes:
        format: Output format (json, csv, txt, etc.)
        path: Output file path
        mode: Write mode ('w' or 'a')
    """

    def __init__(
        self,
        format: str = "json",
        path: str | None = None,
        mode: str = "w"
    ) -> None:
        super().__init__(name="save")
        self.format = format
        self.path = path
        self.mode = mode

    def execute(self, context: PipelineContext) -> Any:
        """Execute the save stage.

        Args:
            context: Pipeline context

        Returns:
            Path to saved file or saved data
        """
        data = context.parsed_data or context.data
        if data is None:
            raise PipelineStageError(
                stage=self.name,
                message="No data to save. Run parse stage first."
            )

        import json

        # Handle different formats
        if self.format == "json":
            output = self._to_json(data)
        elif self.format == "txt":
            output = self._to_text(data)
        elif self.format == "csv":
            output = self._to_csv(data)
        else:
            output = str(data)

        # Save to file or return
        if self.path:
            with open(self.path, self.mode, encoding="utf-8") as f:
                f.write(output)
            return self.path

        return output

    def _to_json(self, data: Any) -> str:
        """Convert data to JSON string."""
        import json

        # Handle Selectors objects
        if hasattr(data, "text"):
            data = data.text()
        elif hasattr(data, "all"):
            # It's a Selectors, get all text
            try:
                data = data.all()
                if hasattr(data, "text"):
                    data = [d.text() for d in data]
            except Exception:
                pass

        return json.dumps(data, indent=2, default=str, ensure_ascii=False)

    def _to_text(self, data: Any) -> str:
        """Convert data to text string."""
        if hasattr(data, "text"):
            return data.text()
        elif hasattr(data, "all"):
            try:
                items = data.all()
                return "\n".join(
                    item.text() if hasattr(item, "text") else str(item)
                    for item in items
                )
            except Exception:
                return str(data)
        return str(data)

    def _to_csv(self, data: Any) -> str:
        """Convert data to CSV string."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Handle Selectors
        if hasattr(data, "all"):
            try:
                items = data.all()
                for item in items:
                    row = [item.text() if hasattr(item, "text") else str(item)]
                    writer.writerow(row)
            except Exception:
                writer.writerow([str(data)])
        else:
            writer.writerow([str(data)])

        return output.getvalue()


class Pipeline:
    """Main Pipeline class for fetch-parse-save workflows.

    This class provides a chainable API for building and executing
    web scraping pipelines.

    Example:
        >>> pipeline = Pipeline()
        >>> result = pipeline.fetch(url).parse(".title").save(format='json')
        >>> print(result.data)

        >>> # Or with custom stages
        >>> pipeline = Pipeline()
        >>> pipeline.add_stage(FetchStage(url="https://example.com"))
        >>> pipeline.add_stage(ParseStage(selector="h1"))
        >>> result = pipeline.run()

    Attributes:
        context: Pipeline execution context
        stages: List of stages in order
    """

    def __init__(self) -> None:
        self.context = PipelineContext()
        self._stages: list[PipelineStage] = []
        self._current_stage: PipelineStage | None = None

    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        """Add a stage to the pipeline.

        Args:
            stage: Stage to add

        Returns:
            Self for chaining
        """
        self._stages.append(stage)
        return self

    def fetch(
        self,
        url: str | None = None,
        fetcher: Any = None,
        **kwargs: Any
    ) -> "Pipeline":
        """Add a fetch stage to the pipeline.

        Args:
            url: URL to fetch
            fetcher: Optional custom fetcher
            **kwargs: Additional fetcher arguments

        Returns:
            Self for chaining
        """
        stage = FetchStage(url=url, fetcher=fetcher, **kwargs)
        self._stages.append(stage)
        self._current_stage = stage
        return self

    def parse(
        self,
        selector: str | None = None,
        extract_all: bool = False
    ) -> "Pipeline":
        """Add a parse stage to the pipeline.

        Args:
            selector: CSS selector to extract
            extract_all: Whether to extract all matches

        Returns:
            Self for chaining
        """
        stage = ParseStage(selector=selector, extract_all=extract_all)
        self._stages.append(stage)
        self._current_stage = stage
        return self

    def save(
        self,
        format: str = "json",
        path: str | None = None
    ) -> "Pipeline":
        """Add a save stage to the pipeline.

        Args:
            format: Output format (json, csv, txt)
            path: Output file path

        Returns:
            Self for chaining
        """
        stage = SaveStage(format=format, path=path)
        self._stages.append(stage)
        self._current_stage = stage
        return self

    def before(self, hook: StageHookWithoutArg) -> "Pipeline":
        """Add a before hook to the current stage.

        Args:
            hook: Function to run before the current stage

        Returns:
            Self for chaining
        """
        if self._current_stage:
            self._current_stage.before(hook)
        return self

    def after(self, hook: StageHook) -> "Pipeline":
        """Add an after hook to the current stage.

        Args:
            hook: Function to run after the current stage

        Returns:
            Self for chaining
        """
        if self._current_stage:
            self._current_stage.after(hook)
        return self

    def run(self) -> PipelineResult:
        """Execute the pipeline.

        Returns:
            PipelineResult with data, metadata, and errors
        """
        result = PipelineResult()
        result.metadata.start_time = datetime.now()

        try:
            # Run each stage
            for stage in self._stages:
                if not stage.enabled:
                    continue

                stage.run(self.context)

            # Pipeline completed
            result.data = self.context.parsed_data or self.context.data
            result.metadata.end_time = datetime.now()
            result.stage_results = dict(self.context._stage_outputs)

        except PipelineStageError as e:
            # Stage failed - add error and mark complete
            result.add_error(
                stage=e.stage,
                message=e.message,
                exception=e.original
            )
            result.metadata.end_time = datetime.now()
            result.stage_results = dict(self.context._stage_outputs)

        except Exception as e:
            # Unexpected error
            result.add_error(
                stage="pipeline",
                message=str(e)
            )
            result.metadata.end_time = datetime.now()
            result.stage_results = dict(self.context._stage_outputs)

        return result


__all__ = [
    "Pipeline",
    "PipelineStage",
    "FetchStage",
    "ParseStage",
    "SaveStage",
    "PipelineResult",
    "PipelineMetadata",
    "PipelineError",
    "PipelineContext",
    "PipelineStageError",
    "StageStatus",
]
