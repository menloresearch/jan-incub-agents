from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import PyPDF2
import io


class ProcessorBase(ABC):
    """Base class for document processors."""
    
    def __init__(self):
        pass

    @abstractmethod
    def file_to_text(self, file: Path) -> str:
        """Convert file to text."""
        pass


class PyPDF2Processor(ProcessorBase):
    """Processor using PyPDF2 to extract text from PDF files."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def file_to_text(self, file: Path) -> str:
        """Extract text from PDF using PyPDF2."""
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")
        
        if file.suffix.lower() != '.pdf':
            raise ValueError(f"PyPDF2Processor only supports PDF files, got: {file.suffix}")
        
        text_content = []
        
        with open(file, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content.append(page.extract_text())
        
        return '\n'.join(text_content)


class MarkItDownProcessor(ProcessorBase):
    """Processor using MarkItDown to extract text from various file formats."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from markitdown import MarkItDown
            self.markitdown = MarkItDown()
        except ImportError:
            raise ImportError("MarkItDown library not installed. Install with: pip install markitdown")
    
    def file_to_text(self, file: Path) -> str:
        """Extract text from various file formats using MarkItDown."""
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")
        
        result = self.markitdown.convert(str(file))
        return result.text_content if hasattr(result, 'text_content') else str(result)


class DoclingProcessor(ProcessorBase):
    """Processor using Docling to extract text from various file formats."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()
        except ImportError:
            raise ImportError("Docling library not installed. Install with: pip install docling")
    
    def file_to_text(self, file: Path) -> str:
        """Extract text from various file formats using Docling."""
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")
        
        result = self.converter.convert(str(file))
        return result.document.export_to_text()


class MarkerProcessor(ProcessorBase):
    """Processor using Marker to convert PDF to markdown with high accuracy."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
            
            # Initialize converter with model artifacts
            self.converter = PdfConverter(
                artifact_dict=create_model_dict(),
            )
            self.text_from_rendered = text_from_rendered
        except ImportError:
            raise ImportError("Marker library not installed. Install with: pip install marker-pdf")
    
    def file_to_text(self, file: Path) -> str:
        """Extract text from PDF using Marker."""
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")
        
        if file.suffix.lower() != '.pdf':
            raise ValueError(f"MarkerProcessor only supports PDF files, got: {file.suffix}")
        
        # Convert PDF to markdown using Marker's new API
        rendered = self.converter(str(file))
        text, _, images = self.text_from_rendered(rendered)
        
        return text


class NativeProcessor(ProcessorBase):
    """Processor that returns file content directly for native LLM processing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def file_to_text(self, file: Path) -> str:
        """Return file path for native LLM processing."""
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")
        
        # For native processing, we return the file path
        # The actual LLM API will handle the file directly
        return str(file)