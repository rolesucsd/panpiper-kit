import sys
import pathlib
from unittest.mock import patch, MagicMock

# Ensure project src/ is on sys.path for imports when running tests locally
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from panpiper_kit.external import require


@patch('shutil.which')
def test_require_success(mock_which):
    """Test require function with existing command."""
    mock_which.return_value = '/usr/bin/test_command'
    
    result = require('test_command')
    assert result == '/usr/bin/test_command'
    mock_which.assert_called_once_with('test_command')


@patch('shutil.which')
def test_require_not_found(mock_which):
    """Test require function with non-existent command."""
    mock_which.return_value = None
    
    try:
        require('nonexistent_command')
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Required executable 'nonexistent_command' not found in PATH" in str(e)
