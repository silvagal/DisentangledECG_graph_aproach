# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Script para agregar resultados de múltiplas execuções de suítes de experimentos.

Este script coleta os resultados de todas as execuções de uma suíte,
calcula estatísticas (média, desvio padrão) e gera relatórios consolidados.

Uso:
    # Agregar resultados de uma suíte específica
    python aggregate_suite_results.py part1_arch
    
    # Especificar diretório personalizado
    python aggregate_suite_results.py part1_arch --input-dir output_suite_runs
    
    # Gerar apenas o relatório, sem salvar CSVs
    python aggregate_suite_results.py part1_arch --report-only
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys

class SuiteResultsAggregator:
    """Agregador de resultados de múltiplas execuções de suítes."""
    
    def __init__(self, suite_name: str, input_dir: str = "output_suite_runs"):
        self.suite_name = suite_name
        self.input_dir = Path(input_dir)
        self.suite_pattern = f"{suite_name}_run_*"
    
    def find_suite_runs(self) -> List[Path]:
        """Encontra todos os diretórios de execução da suíte."""
        pattern = f"{self.suite_name}_run_*"
        runs = list(self.input_dir.glob(pattern))
        return sorted(runs)
    
    def load_run_results(self, run_dir: Path) -> Optional[Dict]:
        """Carrega os resultados de uma execução específica."""
        metadata_file = run_dir / "run_metadata.json"
        
        if not metadata_file.exists():
            print(f"⚠️  Metadata não encontrada em {run_dir}")
            return None
        
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"❌ Erro ao ler metadata de {run_dir}: {e}")
            return None
        
        if not metadata.get("completed", False):
            print(f"⚠️  Execução {run_dir.name} não foi completada")
            return None
        
        # Coletar métricas de cada experimento
        results = {
            "run_id": metadata.get("run_id"),
            "seed": metadata.get("seed"),
            "experiments": {}
        }
        
        output_dir = run_dir / "output"
        if not output_dir.exists():
            print(f"⚠️  Diretório output não encontrado em {run_dir}")
            return None
        
        for exp_dir in output_dir.iterdir():
            if exp_dir.is_dir():
                metrics_file = exp_dir / "metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file) as f:
                            metrics = json.load(f)
                        results["experiments"][exp_dir.name] = metrics
                    except Exception as e:
                        print(f"⚠️  Erro ao ler métricas de {exp_dir}: {e}")
        
        return results
    
    def aggregate_results(self) -> Dict:
        """Agrega resultados de todas as execuções."""
        runs = self.find_suite_runs()
        
        if not runs:
            raise ValueError(f"Nenhuma execução encontrada para a suíte '{self.suite_name}' em {self.input_dir}")
        
        print(f"📊 Agregando resultados de {len(runs)} execuções da suíte '{self.suite_name}'")
        
        all_results = []
        experiment_names = set()
        
        # Carregar resultados de cada execução
        for run_dir in runs:
            print(f"   Processando {run_dir.name}...")
            run_results = self.load_run_results(run_dir)
            
            if run_results:
                all_results.append(run_results)
                experiment_names.update(run_results["experiments"].keys())
        
        if not all_results:
            raise ValueError("Nenhum resultado válido encontrado")
        
        print(f"✅ {len(all_results)} execuções válidas encontradas")
        print(f"📋 Experimentos encontrados: {sorted(experiment_names)}")
        
        # Criar DataFrame com todos os resultados
        rows = []
        for run_result in all_results:
            for exp_name, metrics in run_result["experiments"].items():
                row = {
                    "run_id": run_result["run_id"],
                    "seed": run_result["seed"],
                    "experiment": exp_name,
                    **metrics
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Calcular estatísticas agregadas por experimento
        metrics_columns = [col for col in df.columns if col not in ["run_id", "seed", "experiment"]]
        
        stats = []
        for exp_name in sorted(experiment_names):
            exp_data = df[df["experiment"] == exp_name]
            
            if len(exp_data) == 0:
                continue
            
            row = {"experiment": exp_name}
            
            for metric in metrics_columns:
                if metric in exp_data.columns:
                    values = pd.to_numeric(exp_data[metric], errors='coerce').dropna()
                    
                    if len(values) > 0:
                        row[f"{metric}_mean"] = values.mean()
                        row[f"{metric}_std"] = values.std()
                        row[f"{metric}_min"] = values.min()
                        row[f"{metric}_max"] = values.max()
                        row[f"{metric}_count"] = len(values)
                    else:
                        row[f"{metric}_mean"] = np.nan
                        row[f"{metric}_std"] = np.nan
                        row[f"{metric}_min"] = np.nan
                        row[f"{metric}_max"] = np.nan
                        row[f"{metric}_count"] = 0
            
            stats.append(row)
        
        stats_df = pd.DataFrame(stats)
        
        return {
            "suite_name": self.suite_name,
            "total_runs": len(all_results),
            "total_experiments": len(experiment_names),
            "raw_results": df,
            "aggregated_stats": stats_df,
            "run_directories": [str(r) for r in runs]
        }
    
    def save_results(self, results: Dict, output_dir: Optional[str] = None):
        """Salva os resultados agregados em arquivos."""
        if output_dir is None:
            output_dir = f"aggregated_results_{self.suite_name}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Salvar dados brutos
        raw_file = output_path / f"{self.suite_name}_raw_results.csv"
        results["raw_results"].to_csv(raw_file, index=False)
        print(f"💾 Resultados brutos salvos em: {raw_file}")
        
        # Salvar estatísticas agregadas
        stats_file = output_path / f"{self.suite_name}_aggregated_stats.csv"
        results["aggregated_stats"].to_csv(stats_file, index=False)
        print(f"📊 Estatísticas agregadas salvas em: {stats_file}")
        
        # Salvar metadata
        metadata = {
            "suite_name": results["suite_name"],
            "total_runs": results["total_runs"],
            "total_experiments": results["total_experiments"],
            "run_directories": results["run_directories"],
            "files_generated": {
                "raw_results": str(raw_file),
                "aggregated_stats": str(stats_file)
            }
        }
        
        metadata_file = output_path / f"{self.suite_name}_aggregation_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📝 Metadata salva em: {metadata_file}")
        
        return output_path
    
    def generate_report(self, results: Dict) -> str:
        """Gera um relatório textual dos resultados."""
        report_lines = [
            f"RELATÓRIO DE AGREGAÇÃO - SUÍTE: {results['suite_name'].upper()}",
            "=" * 80,
            f"Total de execuções: {results['total_runs']}",
            f"Total de experimentos: {results['total_experiments']}",
            "",
            "ESTATÍSTICAS PRINCIPAIS (Accuracy):",
            "-" * 40
        ]
        
        # Mostrar estatísticas de accuracy
        stats_df = results["aggregated_stats"]
        if "accuracy_mean" in stats_df.columns:
            for _, row in stats_df.iterrows():
                exp_name = row["experiment"]
                acc_mean = row.get("accuracy_mean", np.nan)
                acc_std = row.get("accuracy_std", np.nan)
                
                if not pd.isna(acc_mean):
                    report_lines.append(
                        f"{exp_name:<30} {acc_mean:6.2f}% ± {acc_std:5.2f}%"
                    )
        
        # Adicionar melhor experimento
        if "accuracy_mean" in stats_df.columns:
            best_exp = stats_df.loc[stats_df["accuracy_mean"].idxmax()]
            report_lines.extend([
                "",
                "MELHOR EXPERIMENTO:",
                "-" * 20,
                f"Nome: {best_exp['experiment']}",
                f"Accuracy: {best_exp['accuracy_mean']:.2f}% ± {best_exp['accuracy_std']:.2f}%"
            ])
        
        report_lines.extend([
            "",
            "=" * 80,
            f"Dados brutos: {results['total_runs'] * results['total_experiments']} pontos",
            "Para análise detalhada, consulte os arquivos CSV gerados."
        ])
        
        return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Agrega resultados de múltiplas execuções de suítes de experimentos",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "suite_name",
        help="Nome da suíte para agregar resultados"
    )
    
    parser.add_argument(
        "--input-dir",
        default="output_suite_runs",
        help="Diretório onde estão as execuções (padrão: output_suite_runs)"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Diretório para salvar resultados agregados (padrão: aggregated_results_<suite_name>)"
    )
    
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Apenas gera relatório, não salva arquivos CSV"
    )
    
    args = parser.parse_args()
    
    try:
        # Criar agregador
        aggregator = SuiteResultsAggregator(args.suite_name, args.input_dir)
        
        # Agregar resultados
        print(f"🔍 Procurando execuções da suíte '{args.suite_name}' em {args.input_dir}")
        results = aggregator.aggregate_results()
        
        # Gerar relatório
        report = aggregator.generate_report(results)
        print("\n" + report)
        
        # Salvar arquivos (se solicitado)
        if not args.report_only:
            output_dir = aggregator.save_results(results, args.output_dir)
            print(f"\n✅ Agregação concluída. Arquivos salvos em: {output_dir}")
        else:
            print("\n💡 Para salvar os resultados, execute novamente sem --report-only")
            
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
