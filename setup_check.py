"""
Verifica la instalacion y el entorno de CONFORMER-SER.

Comprueba version de Python, paquetes requeridos, disponibilidad de CUDA,
estructura del proyecto y ruta del dataset IEMOCAP.

Uso:
    python setup_check.py

Funciona desde cualquier directorio: todas las rutas se resuelven contra la
raiz del repositorio (el directorio que contiene este archivo).
"""

import sys
import os
import multiprocessing

# Raiz del repositorio: este archivo vive en la raiz.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def check_python_version():
    """Check Python version."""
    print("="*60)
    print("Checking Python version...")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[X] Python 3.8+ required")
        return False
    else:
        print("[OK] Python version OK")
        return True


def check_imports():
    """Check if required packages are installed."""
    print("\n" + "="*60)
    print("Checking required packages...")
    
    packages = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
        'transformers': 'Transformers',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tensorboard': 'TensorBoard'
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"[OK] {name:20s} installed")
        except ImportError:
            print(f"[X]  {name:20s} NOT installed")
            all_ok = False

    # Extras opcionales: imports protegidos en el codigo. Su ausencia no es un
    # error, pero condiciona que estudios se pueden ejecutar.
    optional = {
        'funasr': 'funasr (Emotion2Vec: estudios 50 y 57)',
        'opensmile': 'opensmile (eGeMAPS: estudio 38)',
        'bitsandbytes': 'bitsandbytes (optimizadores 8-bit)',
        'xformers': 'xformers (memory-efficient attention)',
        'triton': 'triton (kernels compilados)',
    }
    print("")
    print("Extras opcionales:")
    for package, name in optional.items():
        try:
            __import__(package)
            print(f"[OK] {name}")
        except Exception:
            print(f"[--] {name} - no instalado")

    return all_ok


def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    print("\n" + "="*60)
    print("Checking PyTorch...")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"[OK] CUDA available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("[!] CUDA not available (will use CPU)")
        
        return True
    except Exception as e:
        print(f"[X] Error checking PyTorch: {e}")
        return False


def check_transformers():
    """Check Transformers library."""
    print("\n" + "="*60)
    print("Checking Transformers...")
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        
        # Try loading a tokenizer
        print("Testing tokenizer loading...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        print("[OK] Tokenizer loading works")
        
        return True
    except Exception as e:
        print(f"[X] Error checking Transformers: {e}")
        return False


def check_dataset_path():
    """Check if IEMOCAP dataset path exists."""
    print("\n" + "="*60)
    print("Checking dataset path...")
    
    config_path = os.path.join(REPO_ROOT, "config", "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"[!] Config file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        data_path = config['dataset']['data_path']
        print(f"Dataset path from config: {data_path}")
        
        if os.path.exists(data_path):
            print(f"[OK] Dataset path exists")
            
            # Check for Session directories
            sessions_found = []
            for i in range(1, 9):
                session_path = os.path.join(data_path, f"Session{i}")
                if os.path.exists(session_path):
                    sessions_found.append(i)
            
            if sessions_found:
                print(f"  Found sessions: {sessions_found}")
                return True
            else:
                print(f"[!] No Session directories found in {data_path}")
                return False
        else:
            print(f"[X] Dataset path does not exist: {data_path}")
            print(f"   Please update the path in {config_path}")
            return False
    
    except Exception as e:
        print(f"[X] Error checking dataset: {e}")
        return False


def check_hardware():
    """Check hardware and recommend settings."""
    print("\n" + "="*60)
    print("Checking hardware and recommending settings...")
    
    try:
        num_cores = multiprocessing.cpu_count()
        print(f"Number of CPU cores: {num_cores}")
        
        recommended_workers = min(8, num_cores)
        print(f"  - Recommended 'num_workers' in config.yaml: {recommended_workers}")
        
        return True
    except Exception as e:
        print(f"[X] Error checking hardware: {e}")
        return False


def check_project_structure():
    """Check if project structure is correct."""
    print("\n" + "="*60)
    print("Checking project structure...")
    
    required_dirs = [
        'config',
        'config/ablation',
        'src',
        'src/data',
        'src/models',
        'src/ablation_studies',
        'src/utils',
        'scripts',
        'scripts/windows',
        'docs',
        'notebooks',
    ]

    required_files = [
        # Configuracion
        'config/config.yaml',
        'config/ablation/ABLATION_SUMMARY.md',
        # Paquete src
        'src/__init__.py',
        'src/data/__init__.py',
        'src/data/dataset.py',
        'src/data/preprocessing.py',
        'src/models/__init__.py',
        'src/models/conformer.py',
        'src/models/fusion.py',
        'src/ablation_studies/__init__.py',
        'src/ablation_studies/classification_heads.py',
        'src/utils/__init__.py',
        'src/utils/losses.py',
        'src/utils/metrics.py',
        'src/utils/logger.py',
        # Puntos de entrada
        'src/train.py',
        'src/evaluate.py',
        'src/evaluate_ensemble.py',
        'scripts/run_ablation.py',
        'scripts/run_test_only.py',
        # Raiz
        'requirements.txt',
        'README.md',
        'AGENTS.md',
    ]

    all_ok = True

    for dir_path in required_dirs:
        if os.path.isdir(os.path.join(REPO_ROOT, dir_path)):
            print(f"[OK] Directory exists: {dir_path}")
        else:
            print(f"[X]  Directory missing: {dir_path}")
            all_ok = False

    for file_path in required_files:
        if os.path.isfile(os.path.join(REPO_ROOT, file_path)):
            print(f"[OK] File exists: {file_path}")
        else:
            print(f"[X]  File missing: {file_path}")
            all_ok = False

    return all_ok


def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("CONFORMER-SER SETUP VERIFICATION")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_imports),
        ("PyTorch", check_pytorch),
        ("Transformers", check_transformers),
        ("Hardware", check_hardware),
        ("Project Structure", check_project_structure),
        ("Dataset Path", check_dataset_path)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"[X] Error in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "[OK] PASS" if passed else "[X]  FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("[OK] All checks passed! You're ready to train.")
        print("")
        print("Siguientes pasos:")
        print("  1. Revisa config\\config.yaml (dataset.data_path)")
        print("  2. Linea base : python src\\train.py --config config\\config.yaml --device cuda")
        print("  3. Ablacion   : python scripts\\run_ablation.py --study 45 --device cuda")
        print("                  (estudios validos: 1-50 y 52-63; no existe el 51)")
        print("  4. Monitoreo  : tensorboard --logdir=logs")
        print("  5. Guia corta : quickstart.md")
    else:
        print("[X] Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Update dataset path in config/config.yaml")
        print("  - Ensure CUDA is properly installed for GPU support")
    print("="*60)


if __name__ == '__main__':
    main()

