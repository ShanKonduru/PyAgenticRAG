"""
Enhanced data source connectors for PyAgenticRAG
Supports multiple data source types: files, databases, web APIs, etc.
"""
import logging
import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
import requests
from urllib.parse import urljoin, urlparse

# Document processing imports
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQL_AVAILABLE = True
except ImportError:
    SQL_AVAILABLE = False

from src.vector_store import Document, create_documents_from_text

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    def load_documents(self) -> List[Document]:
        """Load documents from the data source"""
        pass
    
    @abstractmethod
    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the data source"""
        pass

class FileDataSource(DataSource):
    """Data source for local files (PDF, DOCX, TXT, etc.)"""
    
    def __init__(self, data_path: str, file_extensions: Optional[List[str]] = None):
        self.data_path = Path(data_path)
        self.file_extensions = file_extensions or ['.txt', '.pdf', '.docx', '.md', '.json']
        
        if not self.data_path.exists():
            logger.warning(f"Data path does not exist: {data_path}")
    
    def load_documents(self) -> List[Document]:
        """Load documents from files"""
        documents = []
        
        if not self.data_path.exists():
            logger.warning(f"Data path {self.data_path} does not exist")
            return documents
        
        # Get all files with supported extensions
        files = []
        if self.data_path.is_file():
            files = [self.data_path]
        else:
            for ext in self.file_extensions:
                files.extend(self.data_path.glob(f"**/*{ext}"))
        
        for file_path in files:
            try:
                content = self._extract_text_from_file(file_path)
                if content:
                    # Create metadata
                    metadata = {
                        "source_type": "file",
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "file_extension": file_path.suffix,
                        "file_size": file_path.stat().st_size,
                        "modified_time": file_path.stat().st_mtime
                    }
                    
                    # Create chunked documents
                    file_documents = create_documents_from_text(
                        text=content,
                        source=str(file_path),
                        metadata=metadata
                    )
                    documents.extend(file_documents)
                    
                    logger.info(f"Loaded {len(file_documents)} chunks from {file_path}")
                    
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
        
        return documents
    
    def _extract_text_from_file(self, file_path: Path) -> str:
        """Extract text content from a file based on its extension"""
        extension = file_path.suffix.lower()
        
        if extension == '.txt' or extension == '.md':
            return file_path.read_text(encoding='utf-8')
        
        elif extension == '.pdf':
            return self._extract_from_pdf(file_path)
        
        elif extension == '.docx':
            return self._extract_from_docx(file_path)
        
        elif extension == '.json':
            return self._extract_from_json(file_path)
        
        else:
            logger.warning(f"Unsupported file extension: {extension}")
            return ""
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            logger.error("PyPDF2 not installed. Cannot process PDF files.")
            return ""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            logger.error("python-docx not installed. Cannot process DOCX files.")
            return ""
        
        try:
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""
    
    def _extract_from_json(self, file_path: Path) -> str:
        """Extract text from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return json.dumps(data, indent=2)
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            return ""
    
    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the file data source"""
        return {
            "type": "file_system",
            "path": str(self.data_path),
            "supported_extensions": self.file_extensions,
            "exists": self.data_path.exists()
        }

class WebDataSource(DataSource):
    """Data source for web content (URLs, web scraping)"""
    
    def __init__(self, urls: List[str], max_depth: int = 1):
        self.urls = urls
        self.max_depth = max_depth
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PyAgenticRAG/1.0 (Data Collection Bot)'
        })
    
    def load_documents(self) -> List[Document]:
        """Load documents from web URLs"""
        documents = []
        
        for url in self.urls:
            try:
                content = self._fetch_web_content(url)
                if content:
                    metadata = {
                        "source_type": "web",
                        "url": url,
                        "domain": urlparse(url).netloc
                    }
                    
                    # Create chunked documents
                    web_documents = create_documents_from_text(
                        text=content,
                        source=url,
                        metadata=metadata
                    )
                    documents.extend(web_documents)
                    
                    logger.info(f"Loaded {len(web_documents)} chunks from {url}")
                    
            except Exception as e:
                logger.error(f"Error loading URL {url}: {e}")
        
        return documents
    
    def _fetch_web_content(self, url: str) -> str:
        """Fetch and extract text content from a web URL"""
        if not BS4_AVAILABLE:
            logger.error("BeautifulSoup4 not installed. Cannot process web content.")
            return ""
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return ""
    
    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the web data source"""
        return {
            "type": "web_scraping",
            "urls": self.urls,
            "max_depth": self.max_depth
        }

class DatabaseDataSource(DataSource):
    """Data source for SQL databases"""
    
    def __init__(self, connection_string: str, query: str, text_column: str = "content"):
        if not SQL_AVAILABLE:
            raise ImportError("SQLAlchemy not installed. Cannot use database data source.")
        
        self.connection_string = connection_string
        self.query = query
        self.text_column = text_column
        self.engine = create_engine(connection_string)
    
    def load_documents(self) -> List[Document]:
        """Load documents from database query results"""
        documents = []
        
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(self.query))
                rows = result.fetchall()
                
                for i, row in enumerate(rows):
                    row_dict = row._asdict() if hasattr(row, '_asdict') else dict(row)
                    
                    # Extract text content
                    content = str(row_dict.get(self.text_column, ""))
                    
                    if content:
                        metadata = {
                            "source_type": "database",
                            "row_index": i,
                            **{k: v for k, v in row_dict.items() if k != self.text_column}
                        }
                        
                        # Create chunked documents
                        db_documents = create_documents_from_text(
                            text=content,
                            source=f"db_row_{i}",
                            metadata=metadata
                        )
                        documents.extend(db_documents)
                
                logger.info(f"Loaded {len(documents)} chunks from database query")
                
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
        
        return documents
    
    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the database data source"""
        return {
            "type": "database",
            "connection_string": self.connection_string.replace(
                self.connection_string.split('@')[0] if '@' in self.connection_string else '',
                "***"
            ),  # Hide credentials
            "query": self.query,
            "text_column": self.text_column
        }

class APIDataSource(DataSource):
    """Data source for REST APIs"""
    
    def __init__(self, base_url: str, endpoints: List[str], headers: Optional[Dict[str, str]] = None):
        self.base_url = base_url
        self.endpoints = endpoints
        self.headers = headers or {}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def load_documents(self) -> List[Document]:
        """Load documents from API endpoints"""
        documents = []
        
        for endpoint in self.endpoints:
            try:
                url = urljoin(self.base_url, endpoint)
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Try to parse as JSON first, then as text
                try:
                    data = response.json()
                    content = json.dumps(data, indent=2)
                except json.JSONDecodeError:
                    content = response.text
                
                if content:
                    metadata = {
                        "source_type": "api",
                        "url": url,
                        "endpoint": endpoint,
                        "status_code": response.status_code
                    }
                    
                    # Create chunked documents
                    api_documents = create_documents_from_text(
                        text=content,
                        source=url,
                        metadata=metadata
                    )
                    documents.extend(api_documents)
                    
                    logger.info(f"Loaded {len(api_documents)} chunks from {url}")
                    
            except Exception as e:
                logger.error(f"Error loading from API endpoint {endpoint}: {e}")
        
        return documents
    
    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the API data source"""
        return {
            "type": "rest_api",
            "base_url": self.base_url,
            "endpoints": self.endpoints
        }

class DataSourceManager:
    """Manager for handling multiple data sources"""
    
    def __init__(self):
        self.data_sources: List[DataSource] = []
    
    def add_file_source(self, data_path: str, file_extensions: Optional[List[str]] = None):
        """Add a file data source"""
        source = FileDataSource(data_path, file_extensions)
        self.data_sources.append(source)
        logger.info(f"Added file data source: {data_path}")
    
    def add_web_source(self, urls: List[str], max_depth: int = 1):
        """Add a web data source"""
        source = WebDataSource(urls, max_depth)
        self.data_sources.append(source)
        logger.info(f"Added web data source with {len(urls)} URLs")
    
    def add_database_source(self, connection_string: str, query: str, text_column: str = "content"):
        """Add a database data source"""
        source = DatabaseDataSource(connection_string, query, text_column)
        self.data_sources.append(source)
        logger.info("Added database data source")
    
    def add_api_source(self, base_url: str, endpoints: List[str], headers: Optional[Dict[str, str]] = None):
        """Add an API data source"""
        source = APIDataSource(base_url, endpoints, headers)
        self.data_sources.append(source)
        logger.info(f"Added API data source: {base_url}")
    
    def load_all_documents(self) -> List[Document]:
        """Load documents from all configured data sources"""
        all_documents = []
        
        for source in self.data_sources:
            try:
                documents = source.load_documents()
                all_documents.extend(documents)
                logger.info(f"Loaded {len(documents)} documents from {source.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error loading from {source.__class__.__name__}: {e}")
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def get_sources_info(self) -> List[Dict[str, Any]]:
        """Get information about all configured data sources"""
        return [source.get_source_info() for source in self.data_sources]