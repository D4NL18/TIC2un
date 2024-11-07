import { Component, ViewChild } from '@angular/core';
import { NfResultsComponent } from '../../components/nf-results/nf-results.component';
import { ButtonComponent } from '../../components/button/button.component';
import { NavComponent } from '../../components/nav/nav.component';

@Component({
  selector: 'app-nf',
  standalone: true,
  imports: [ButtonComponent, NfResultsComponent, NavComponent],
  templateUrl: './nf.component.html',
  styleUrl: './nf.component.scss'
})
export class NfComponent {
  @ViewChild(NfResultsComponent) nfResultsComponent!: NfResultsComponent

  onTrainNF() {
    if(this.nfResultsComponent) {
      this.nfResultsComponent.trainNF()
    }
  }
}
