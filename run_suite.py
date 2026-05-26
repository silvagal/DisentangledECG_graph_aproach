# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Script orquestrador para executar suítes completas de experimentos.

Este script permite executar uma suíte de experimentos 5 vezes com seeds diferentes,
salvando os resultados em diretórios separados para agregação posterior.

Uso:
    # Listar todas as suítes disponíveis
    python run_suite.py --list
    
    # Executar uma suíte específica
    python run_suite.py part1_arch
    
    # Executar com configurações personalizadas
    python run_suite.py part1_arch --runs 3 --seeds 42,123,456
    
    # Ver status de execuções anteriores
    python run_suite.py part1_arch --status
"""

import argparse
import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Importar as definições das suítes
try:
    from suites import EXPERIMENT_SUITES, get_suite_names, get_suite_configs, get_suite_info, validate_suite
except ImportError:
    print("❌ ERRO: Arquivo suites.py não encontrado!")
    print("   Certifique-se de que o arquivo suites.py está no mesmo diretório.")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("⚠️  AVISO: PyYAML não encontrado. Funcionalidade limitada.")
    yaml = None

class SuiteRunner:
    """Gerenciador de execução de suítes de experimentos."""
    
    def __init__(self, suite_name: str, base_output_dir: str = "output_suite_runs"):
        self.suite_name = suite_name
        self.base_output_dir = Path(base_output_dir)
        self.suite_configs = get_suite_configs(suite_name)
        
        # Validar suíte
        valid, message = validate_suite(suite_name)
        if not valid:
            raise ValueError(message)
    
    def get_run_directory(self, run_id: int) -> Path:
        """Retorna o diretório para uma execução específica."""
        return self.base_output_dir / f"{self.suite_name}_run_{run_id:02d}"
    
    def get_run_status(self, run_id: int) -> Dict:
        """Verifica o status de uma execução específica."""
        run_dir = self.get_run_directory(run_id)
        status = {
            "run_id": run_id,
            "directory": str(run_dir),
            "exists": run_dir.exists(),
            "completed": False,
            "experiments": [],
            "total_experiments": len(self.suite_configs),
            "completed_experiments": 0,
            "start_time": None,
            "end_time": None,
            "duration": None
        }
        
        if not run_dir.exists():
            return status
        
        # Verificar arquivo de metadata
        metadata_file = run_dir / "run_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                status.update(metadata)
            except Exception as e:
                print(f"⚠️  Erro ao ler metadata de {run_dir}: {e}")
        
        # Verificar status de cada experimento
        for config_path in self.suite_configs:
            exp_name = self._get_experiment_name(config_path)
            exp_dir = run_dir / "output" / exp_name
            exp_status = {
                "config": config_path,
                "name": exp_name,
                "directory": str(exp_dir),
                "completed": (exp_dir / "metrics.json").exists() if exp_dir.exists() else False
            }
            status["experiments"].append(exp_status)
            if exp_status["completed"]:
                status["completed_experiments"] += 1
        
        status["completed"] = status["completed_experiments"] == status["total_experiments"]
        return status
    
    def _get_experiment_name(self, config_path: str) -> str:
        """Extrai o nome do experimento do arquivo de configuração."""
        path = Path(config_path)
        
        if yaml is not None:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if data:
                    return data.get("experiment_name") or data.get("experimentName") or data.get("name") or path.stem
            except Exception:
                pass
        
        # Fallback: usar nome do arquivo
        return path.stem
    
    def prepare_run_directory(self, run_id: int, seed: int) -> Path:
        """Prepara o diretório para uma nova execução."""
        run_dir = self.get_run_directory(run_id)
        
        if run_dir.exists():
            print(f"⚠️  Diretório {run_dir} já existe. Removendo...")
            shutil.rmtree(run_dir)
        
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Criar metadata da execução
        metadata = {
            "suite_name": self.suite_name,
            "run_id": run_id,
            "seed": seed,
            "start_time": datetime.now().isoformat(),
            "configs": self.suite_configs,
            "total_experiments": len(self.suite_configs),
            "completed_experiments": 0,
            "completed": False
        }
        
        with open(run_dir / "run_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return run_dir
    
    def execute_run(self, run_id: int, seed: int, max_concurrent: int = 1, full_dataset: bool = False) -> bool:
        """Executa uma suite completa com um seed específico."""
        print(f"\n🚀 INICIANDO EXECUÇÃO {run_id} DA SUÍTE '{self.suite_name}'")
        print(f"🎲 Seed: {seed}")
        print(f"📋 Experimentos: {len(self.suite_configs)}")
        print(f"⚡ Máximo concorrente: {max_concurrent}")
        
        # Preparar diretório
        run_dir = self.prepare_run_directory(run_id, seed)
        print(f"📁 Diretório: {run_dir}")
        
        # Configurar ambiente
        env = os.environ.copy()
        env.update({
            "PYTHONSEED": str(seed),
            "CUDA_LAUNCH_BLOCKING": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
            "SUITE_OUTPUT_DIR": str(run_dir / "output"),
        })
        
        success = True
        start_time = time.time()
        
        try:
            # Executar usando run_all_experiments.py com configs personalizados
            cmd = [
                sys.executable, "-u", 
                "scripts/run_all_experiments.py",
                "--configs"
            ] + self.suite_configs + [
                "--max-concurrent", str(max_concurrent)
            ]
            if full_dataset:
                cmd.append("--full-dataset")
            
            print(f"🔧 Comando: {' '.join(cmd)}")
            
            # Redirecionar output para o diretório da execução
            log_file = run_dir / "suite_execution.log"
            
            with open(log_file, 'w') as f:
                f.write(f"=== EXECUÇÃO DA SUÍTE {self.suite_name} - RUN {run_id} ===\n")
                f.write(f"Seed: {seed}\n")
                f.write(f"Início: {datetime.now()}\n")
                f.write(f"Comando: {' '.join(cmd)}\n")
                f.write("=" * 80 + "\n\n")
            
            try:
                # Executar o comando
                with open(log_file, 'a') as f:
                    process = subprocess.run(
                        cmd,
                        env=env,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                
                success = process.returncode == 0
                
                if not success:
                    print(f"❌ Execução falhou com código {process.returncode}")
                
            except Exception as e:
                print(f"❌ ERRO AO EXECUTAR COMANDO: {e}")
                success = False
            
        except KeyboardInterrupt:
            print("\n⏹️  EXECUÇÃO INTERROMPIDA PELO USUÁRIO")
            success = False
        except Exception as e:
            print(f"❌ ERRO DURANTE EXECUÇÃO: {e}")
            success = False
        
        # Atualizar metadata
        end_time = time.time()
        duration = end_time - start_time
        
        try:
            with open(run_dir / "run_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            metadata.update({
                "end_time": datetime.now().isoformat(),
                "duration_seconds": duration,
                "success": success,
                "completed": success
            })
            
            # Contar experimentos completados
            completed_count = 0
            for config_path in self.suite_configs:
                exp_name = self._get_experiment_name(config_path)
                exp_dir = run_dir / "output" / exp_name
                if exp_dir.exists() and (exp_dir / "metrics.json").exists():
                    completed_count += 1
            
            metadata["completed_experiments"] = completed_count
            
            with open(run_dir / "run_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Erro ao atualizar metadata: {e}")
        
        if success:
            print(f"✅ EXECUÇÃO {run_id} COMPLETADA COM SUCESSO!")
            print(f"⏱️  Duração: {duration/3600:.1f} horas")
        else:
            print(f"❌ EXECUÇÃO {run_id} FALHOU")
        
        return success
    
    def execute_multiple_runs(self, num_runs: int = 5, seeds: Optional[List[int]] = None, 
                            max_concurrent: int = 1, full_dataset: bool = False) -> Dict:
        """Executa múltiplas execuções da suíte."""
        if seeds is None:
            seeds = [42 + i * 111 for i in range(num_runs)]
        elif len(seeds) != num_runs:
            raise ValueError(f"Número de seeds ({len(seeds)}) deve ser igual ao número de execuções ({num_runs})")
        
        print(f"\n🎯 EXECUTANDO {num_runs} EXECUÇÕES DA SUÍTE '{self.suite_name}'")
        print(f"🎲 Seeds: {seeds}")
        
        results = {
            "suite_name": self.suite_name,
            "total_runs": num_runs,
            "seeds": seeds,
            "start_time": datetime.now().isoformat(),
            "runs": [],
            "successful_runs": 0,
            "failed_runs": 0
        }
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n{'='*80}")
            print(f"EXECUÇÃO {i} de {num_runs}")
            print(f"{'='*80}")
            
            success = self.execute_run(i, seed, max_concurrent, full_dataset=full_dataset)
            
            run_result = {
                "run_id": i,
                "seed": seed,
                "success": success,
                "directory": str(self.get_run_directory(i))
            }
            
            results["runs"].append(run_result)
            
            if success:
                results["successful_runs"] += 1
            else:
                results["failed_runs"] += 1
                print(f"⚠️  Execução {i} falhou. Continuando com as próximas...")
        
        results["end_time"] = datetime.now().isoformat()
        
        # Salvar relatório final
        report_file = self.base_output_dir / f"{self.suite_name}_execution_report.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"🏁 EXECUÇÃO DA SUÍTE '{self.suite_name}' FINALIZADA")
        print(f"✅ Execuções bem-sucedidas: {results['successful_runs']}/{num_runs}")
        print(f"❌ Execuções falhas: {results['failed_runs']}/{num_runs}")
        print(f"📊 Relatório salvo em: {report_file}")
        print(f"{'='*80}")
        
        return results
    
    def print_status(self, num_runs: int = 5):
        """Imprime o status das execuções."""
        print(f"\n📊 STATUS DA SUÍTE '{self.suite_name}'")
        print("=" * 80)
        
        for i in range(1, num_runs + 1):
            status = self.get_run_status(i)
            
            if not status["exists"]:
                icon = "⚪"
                info = "Não executada"
            elif status["completed"]:
                icon = "✅"
                info = f"Completa ({status['completed_experiments']}/{status['total_experiments']} exp.)"
            else:
                icon = "🔄"
                info = f"Em andamento ({status['completed_experiments']}/{status['total_experiments']} exp.)"
            
            print(f"{icon} Execução {i:2d}: {info}")
            
            if status["exists"] and status.get("start_time"):
                print(f"    📁 {status['directory']}")
                if status.get("duration_seconds"):
                    duration = status["duration_seconds"] / 3600
                    print(f"    ⏱️  Duração: {duration:.1f}h")
        
        print("=" * 80)


def list_suites():
    """Lista todas as suítes disponíveis."""
    print("\n📋 SUÍTES DE EXPERIMENTOS DISPONÍVEIS:")
    print("=" * 80)
    
    for suite_name in sorted(get_suite_names()):
        info = get_suite_info(suite_name)
        print(f"\n🔹 {suite_name}")
        print(f"   Configurações: {info['config_count']}")
        print(f"   Total de experimentos: {info['total_experiments']} (com 5 seeds)")
        
        # Mostrar algumas configurações
        configs_to_show = info['configs'][:3]
        for config in configs_to_show:
            print(f"   - {config}")
        
        if info['config_count'] > 3:
            print(f"   ... e mais {info['config_count'] - 3} configurações")
    
    print("\n" + "=" * 80)
    print("💡 Para executar uma suíte: python run_suite.py <nome_da_suite>")
    print("💡 Para ver detalhes: python run_suite.py <nome_da_suite> --status")


def main():
    parser = argparse.ArgumentParser(
        description="Executa suítes completas de experimentos com múltiplos seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Listar todas as suítes
  python run_suite.py --list
  
  # Executar suíte com configuração padrão (5 execuções)
  python run_suite.py part1_arch
  
  # Executar com 3 execuções usando seeds específicos
  python run_suite.py part1_arch --runs 3 --seeds 42,123,456
  
  # Ver status de execuções anteriores
  python run_suite.py part1_arch --status
        """
    )
    
    parser.add_argument(
        "suite_name",
        nargs="?",
        help="Nome da suíte de experimentos para executar"
    )
    
    parser.add_argument(
        "--list",
        action="store_true", 
        help="Lista todas as suítes disponíveis"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Número de execuções (padrão: 5)"
    )
    
    parser.add_argument(
        "--seeds",
        type=str,
        help="Seeds separados por vírgula (ex: 42,123,456). Se não especificado, usa seeds automáticos."
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Máximo de experimentos concorrentes por execução (padrão: 1)"
    )
    
    parser.add_argument(
        "--status", 
        action="store_true",
        help="Mostra o status das execuções da suíte"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_suite_runs",
        help="Diretório base para salvar execuções (padrão: output_suite_runs)"
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Executa sem subamostragem (DS1/DS2 completos).",
    )
    
    args = parser.parse_args()
    
    # Listar suítes
    if args.list:
        list_suites()
        return
    
    # Validar argumentos
    if not args.suite_name:
        parser.error("Nome da suíte é obrigatório (use --list para ver opções)")
    
    # Processar seeds
    seeds = None
    if args.seeds:
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(",")]
        except ValueError:
            parser.error("Seeds devem ser números inteiros separados por vírgula")
    
    try:
        # Criar runner
        runner = SuiteRunner(args.suite_name, args.output_dir)
        
        # Verificar status
        if args.status:
            runner.print_status(args.runs)
            return
        
        # Executar suíte
        results = runner.execute_multiple_runs(
            num_runs=args.runs,
            seeds=seeds,
            max_concurrent=args.max_concurrent,
            full_dataset=args.full_dataset,
        )
        
        # Resumo final
        if results["successful_runs"] == results["total_runs"]:
            print("\n🎉 TODAS AS EXECUÇÕES FORAM BEM-SUCEDIDAS!")
        elif results["successful_runs"] > 0:
            print(f"\n⚠️  {results['failed_runs']} execuções falharam de {results['total_runs']}")
        else:
            print("\n❌ TODAS AS EXECUÇÕES FALHARAM!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
