import { Component, EventEmitter, Output } from '@angular/core';
import { LoadingSpinnerComponent } from '../loading-spinner/loading-spinner.component';
import { CommonModule } from '@angular/common';
import { NeurofuzzyService } from '../../services/neurofuzzy.service';
import { fuzzyService } from '../../services/fuzzy.service';

@Component({
  selector: 'app-nf-results',
  standalone: true,
  imports: [LoadingSpinnerComponent, CommonModule],
  templateUrl: './nf-results.component.html',
  styleUrl: './nf-results.component.scss'
})
export class NfResultsComponent {
  @Output() trainRequested = new EventEmitter<void>()

  imageUrlNF: string | undefined
  imageUrlF: string | undefined
  isLoading: boolean = false;

  constructor(private nfService: NeurofuzzyService, private fService: fuzzyService) { }

  trainNF() {

    this.isLoading = true;

    this.nfService.trainNF().subscribe({
      next: (response) => {
        console.log("Treinamento NF Concluído", response);
        this.fetchImageNF();
      },
      error: (err) => {
        console.log("Erro ao treinar NF:", err);
        this.isLoading = false;
      },
      complete: () => {
        this.isLoading = false;
      }
    });
    this.fService.trainF().subscribe({
      next: (response) => {
        console.log("Treinamento F Concluído", response);
        this.fetchImageF();
      },
      error: (err) => {
        console.log("Erro ao treinar F:", err);
        this.isLoading = false;
      },
      complete: () => {
        this.isLoading = false;
      }
    });
  }

  fetchImageNF() {
    this.nfService.getImage().subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob);
        this.imageUrlNF = url;
      },
      error: (err) => {
        console.error('Erro ao buscar imagem NF:', err);
      }
    });
  }
  fetchImageF() {
    this.fService.getImage().subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob);
        this.imageUrlF = url;
      },
      error: (err) => {
        console.error('Erro ao buscar imagem F:', err);
      }
    });
  }

}
